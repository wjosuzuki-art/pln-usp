"""
ensemble.py -- combina previsões de várias rodadas do train_v2.py.

Cada rodada do train_v2.py salva as probabilidades out-of-fold (OOF) com
os MESMOS folds (GroupKFold determinístico), então dá pra achar os pesos
do ensemble honestamente: os pesos são escolhidos olhando previsões que
nenhum modelo viu no treino.

Opcional: --add_tfidf inclui o TF-IDF+LogReg como membro do ensemble
(sozinho ele não pontua em inovação, mas como peça de um ensemble com
transformers ele costuma ajudar -- erra em lugares diferentes).

Uso:
    python ensemble.py --data train.xlsx --test test.xlsx \
        --runs runs_v2/bertlarge_1790000000 runs_v2/xlmr_1790050000 --add_tfidf
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from data_utils import LABELS, load_train, clean_text
from train_v2 import make_group_folds, metrics_from_probs, write_submission, dup_blend


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="train.xlsx")
    p.add_argument("--test", default=None)
    p.add_argument("--runs", nargs="*", default=[])
    p.add_argument("--add_tfidf", action="store_true")
    p.add_argument("--add_char", action="store_true",
                   help="Inclui também um TF-IDF de CARACTERES (3-5) + LogReg como membro: barato (CPU, ~5 min) "
                        "e é uma 'visão' diferente do texto (pega prefixos, erros de digitação, formatação).")
    p.add_argument("--kfold", type=int, default=3)
    p.add_argument("--step", type=float, default=0.05, help="Resolução da busca de pesos.")
    p.add_argument("--weights", choices=["auto", "grid", "logloss", "mean"], default="auto",
                   help="Como achar os pesos: grid = busca exaustiva pela acurácia (só viável com até 4 "
                        "membros); logloss = ajuste suave por log-loss (serve para qualquer nº de membros, "
                        "menos sujeito a overfitting); mean = média simples; auto = grid se até 4 membros, "
                        "senão logloss.")
    p.add_argument("--dup_blend_alpha", type=float, default=3.0)
    p.add_argument("--out", default="submission_ensemble.xlsx")
    p.add_argument("--tune_bias", action="store_true",
                   help="Ajusta a regra de decisão: soma um viés a c1 e a c5 (c234 fica como "
                        "referência) antes do argmax. Estimativa honesta por cross-fitting.")
    return p.parse_args()


def tfidf_member(df, fold, n_splits, test_texts, kind="word"):
    """Membro clássico do ensemble: TF-IDF + regressão logística, com o MESMO esquema de folds.
    kind='word': palavras 1-2 gramas (o baseline oficial); kind='char': caracteres 3-5."""
    oof = np.zeros((len(df), len(LABELS)), dtype=np.float32)
    if kind == "word":
        mk = lambda: TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        C = 1.0
    else:
        mk = lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True,
                                     max_features=100000)
        C = 1.0
    for k in range(n_splits):
        tr, va = fold != k, fold == k
        vec = mk()
        clf = LogisticRegression(max_iter=1000, C=C).fit(vec.fit_transform(df["text"][tr]), df["label_id"][tr])
        oof[va] = clf.predict_proba(vec.transform(df["text"][va]))
    test = None
    if test_texts is not None:
        vec = mk()
        clf = LogisticRegression(max_iter=1000, C=C).fit(vec.fit_transform(df["text"]), df["label_id"])
        test = clf.predict_proba(vec.transform(test_texts)).astype(np.float32)
    return oof, test


BIAS_GRID = np.round(np.arange(-0.6, 0.61, 0.05), 3)


def biased_pred(P, b1, b5):
    """Soma um viés ao logit de c1 e de c5 (c234 é a referência) antes do argmax."""
    logit = np.log(np.clip(P, 1e-7, 1))
    logit[:, 0] += b1
    logit[:, 2] += b5
    return logit.argmax(1)


def best_bias(P, y):
    return max(((b1, b5) for b1 in BIAS_GRID for b5 in BIAS_GRID),
               key=lambda b: ((biased_pred(P, *b) == y).mean(), -abs(b[0]) - abs(b[1])))


def weight_grid(n, step):
    ticks = np.round(np.arange(0, 1 + 1e-9, step), 6)
    for w in itertools.product(ticks, repeat=n - 1):
        last = 1 - sum(w)
        if last >= -1e-9:
            v = np.array(list(w) + [max(last, 0.0)])
            yield v / v.sum()


def fast_acc_ll(P, y):
    """Acurácia e log-loss em numpy puro (bem mais rápido que o sklearn)."""
    P = P / P.sum(1, keepdims=True)
    return float((P.argmax(1) == y).mean()), float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-7, 1)).mean())


def fit_weights_grid(oofs, y, step, exact=True):
    """Busca exaustiva; critério = acurácia (desempate por log-loss). exact=True usa as métricas
    do sklearn (reproduz exatamente os resultados de antes); exact=False usa a versão rápida."""
    best_w, best_key = None, None
    for w in weight_grid(len(oofs), step):
        P = sum(wi * Pi for wi, Pi in zip(w, oofs))
        if exact:
            m = metrics_from_probs(y, P)
            key = (m["accuracy"], -m["log_loss"])
        else:
            a, l = fast_acc_ll(P, y)
            key = (a, -l)
        if best_key is None or key > best_key:
            best_key, best_w = key, w
    return best_w


def fit_weights_logloss(oofs, y, step=None):
    """Pesos (>=0, somando 1) que minimizam a log-loss da mistura. Problema convexo: rápido e
    estável com qualquer número de modelos."""
    n = len(oofs)
    pt = np.stack([P[np.arange(len(y)), y] for P in oofs], axis=1)  # prob. da classe verdadeira, N x n
    f = lambda w: float(-np.log(np.clip(pt @ w, 1e-7, None)).mean())
    g = lambda w: -(pt / np.clip(pt @ w, 1e-7, None)[:, None]).mean(0)
    res = minimize(f, np.ones(n) / n, jac=g, method="SLSQP", bounds=[(0, 1)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(n)}])
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def fit_weights_mean(oofs, y, step=None):
    return np.ones(len(oofs)) / len(oofs)


def nested_estimate(oofs, y, fold, fit_fn):
    """Estimativa honesta do ensemble: para cada fold, ajusta os pesos SÓ nas linhas dos outros
    folds e avalia nas linhas do fold de fora."""
    P_all = np.zeros_like(oofs[0])
    for k in np.unique(fold):
        te = fold == k
        w = fit_fn([P[~te] for P in oofs], y[~te])
        P_all[te] = sum(wi * P[te] for wi, P in zip(w, oofs))
    return fast_acc_ll(P_all, y)


def main():
    args = parse_args()
    df = load_train(args.data)
    y = df["label_id"].values
    # usa a divisão gravada na primeira rodada (versões diferentes do scikit-learn
    # podem dividir os grupos de forma diferente); sem rodadas, calcula aqui
    if args.runs:
        fold = pd.read_csv(Path(args.runs[0]) / "oof_probs.csv")["fold"].values
        assert len(fold) == len(df), "OOF com número de linhas diferente do train.xlsx"
        args.kfold = int(fold.max()) + 1
    else:
        fold = make_group_folds(df, args.kfold)
    test_texts = None
    if args.test:
        test_texts = pd.read_excel(args.test)["resp_text"].apply(clean_text).tolist()

    names, oofs, tests = [], [], []
    for r in args.runs:
        r = Path(r)
        o = pd.read_csv(r / "oof_probs.csv")
        assert len(o) == len(df), f"{r}: OOF com {len(o)} linhas, esperado {len(df)} (mesmo train.xlsx?)"
        assert (o["clarity"].values == df["clarity"].values).all(), \
            f"{r}: rótulos não batem com o train.xlsx atual -- é o mesmo arquivo?"
        # o esquema de fold desse modelo pode ser diferente do de referência (ex.: uma
        # extração de um treino antigo com 3 folds, junto de rodadas novas com 5) --
        # isso não invalida a média das probabilidades, cada uma continua sendo uma
        # previsão genuinamente fora do treino daquele modelo específico.
        if len(o["fold"].unique()) != args.kfold:
            print(f"[info] {r}: usa {o['fold'].nunique()} folds (referência tem {args.kfold}) -- ok, "
                  f"só entra na média das probabilidades.")
        P = o[[f"p_{l}" for l in LABELS]].values.astype(np.float32)
        if np.isnan(P).any():
            print(f"[aviso] {r} tem folds faltando (rodou com --folds?) -- ignorado.")
            continue
        names.append(r.name)
        oofs.append(P)
        tp = r / "test_probs_mean.npy"
        tests.append(np.load(tp) if tp.exists() else None)

    for flag, kind, label in ((args.add_tfidf, "word", "tfidf_logreg"), (args.add_char, "char", "tfidf_char_logreg")):
        if flag:
            print(f"[info] calculando membro {label} (CPU; alguns minutos)...")
            o, t = tfidf_member(df, fold, args.kfold, test_texts, kind)
            names.append(label)
            oofs.append(o)
            tests.append(t)

    assert oofs, "Nenhum modelo para combinar."
    print("\nModelos individuais (OOF):")
    for n, P in zip(names, oofs):
        m = metrics_from_probs(y, P)
        print(f"  {n:40s} acc={m['accuracy']:.4f} macro-F1={m['macro_f1']:.4f} log-loss={m['log_loss']:.4f}")

    fits = {"mean": fit_weights_mean,
            "logloss": fit_weights_logloss,
            "grid": lambda oo, yy: fit_weights_grid(oo, yy, args.step, exact=False)}
    scores = {}

    def honest(c):
        if c not in scores:
            scores[c] = nested_estimate(oofs, y, fold, fits[c])
        return scores[c]

    if args.weights == "auto":
        # compara os jeitos de pesar pela estimativa HONESTA (pesos ajustados só nos outros folds)
        # e escolhe o de maior acurácia -- a acurácia é a nota; a log-loss não necessariamente a favorece
        cands = ["mean", "logloss"] + (["grid"] if len(oofs) <= 4 else [])
        print("\nComparação dos métodos de pesos (estimativa honesta, pesos ajustados só nos outros folds):")
        for c in cands:
            a, l = honest(c)
            print(f"  {c:8s} acc={a:.4f} log-loss={l:.4f}")
        method = max(cands, key=lambda c: (honest(c)[0], -honest(c)[1]))
        print(f"  -> escolhido: {method}")
    else:
        method = args.weights
        if method == "grid" and len(oofs) > 5:
            print(f"[aviso] grid com {len(oofs)} membros é lento e otimista; prefira --weights auto/logloss.")

    best_w = fit_weights_grid(oofs, y, args.step, exact=True) if method == "grid" else fits[method](oofs, y)
    P = sum(wi * Pi for wi, Pi in zip(best_w, oofs))
    m = metrics_from_probs(y, P)
    print(f"\nPesos do ensemble ({method}):", {n: round(float(w), 2) for n, w in zip(names, best_w)})
    print(f"Ensemble (OOF): acc={m['accuracy']:.4f} macro-F1={m['macro_f1']:.4f} log-loss={m['log_loss']:.4f}")
    if method != "mean":
        n_acc, n_ll = honest(method)
        print(f"  ESTIMATIVA HONESTA (pesos ajustados só nos outros folds): acc={n_acc:.4f} log-loss={n_ll:.4f}"
              f"  <- é esta a que você deve esperar no teste, não o número acima")

    bias = (0.0, 0.0)
    if args.tune_bias:
        # estimativa honesta: vieses escolhidos nos outros folds, avaliados no fold de fora
        pred_cf = np.zeros(len(y), int)
        for k in range(args.kfold):
            out_k = fold == k
            b = best_bias(P[~out_k], y[~out_k])
            pred_cf[out_k] = biased_pred(P[out_k], *b)
        bias = best_bias(P, y)  # vieses finais (todos os dados) -> usados no teste
        print(f"Com ajuste de decisão (cross-fit): acc={(pred_cf == y).mean():.4f} "
              f"| vieses finais c1={bias[0]:+.2f} c5={bias[1]:+.2f}")

    if args.test:
        missing = [n for n, t in zip(names, tests) if t is None]
        if missing:
            print(f"[aviso] sem previsões de teste (test_probs_mean.npy) para: {', '.join(missing)}")
            print("        rode antes, para cada um:  python predict_v2.py --run runs_v2/<pasta> "
                  f"--data {args.data} --test {args.test}")
            print("        submissão não gerada.")
            return
        T = sum(wi * Ti for wi, Ti in zip(best_w, tests))
        if args.tune_bias:  # aplica os vieses convertendo de volta para probabilidades
            L = np.log(np.clip(T, 1e-7, 1)); L[:, 0] += bias[0]; L[:, 2] += bias[1]
            T = np.exp(L) / np.exp(L).sum(1, keepdims=True)
        print("submission (modelo):", write_submission(args.test, T, args.out))
        Tb, hits = dup_blend(test_texts, df, T, args.dup_blend_alpha)
        out_b = args.out.replace(".xlsx", "_dupblend.xlsx")
        print(f"submission (dupblend, {hits} duplicatas):", write_submission(args.test, Tb, out_b))


if __name__ == "__main__":
    main()