"""
members_extra.py -- membros BARATOS e DIFERENTES para o ensemble (só CPU, exceto os embeddings, que usam GPU 1x).

Cada membro sai no mesmo formato das rodadas do train_v2.py (pasta runs_v2/X_<nome>_<timestamp>/ com
oof_probs.csv e test_probs_mean.npy), com os MESMOS folds, e entra direto no ensemble.py e na ablacao.py.

  knn_tfidf  k vizinhos mais parecidos (cosseno sobre TF-IDF de palavras), votos ponderados pela similaridade
  knn_grid   kNN com a config do grid do usuário: TF-IDF unigrama min_df=1, cosseno, k=15, pesos 1/distância
  nb_tfidf   Naive Bayes multinomial sobre TF-IDF
  hgb_stats  gradient boosting sobre ~35 estatísticas do texto (tamanho, frases, números, links, termos
             administrativos...) + 64 componentes SVD do TF-IDF
  knn_emb    k vizinhos mais parecidos sobre embeddings de sentença (cosseno, pesos exp(sim/T))
  lr_emb     regressão logística sobre os mesmos embeddings

Os embeddings (--emb_model, padrão intfloat/multilingual-e5-base, 512 tokens) são calculados UMA vez para
treino+teste (não usam rótulos, então não há vazamento) e guardados em emb_*.npy. Alternativa: --emb_npy com
uma matriz sua (linhas = treino na ordem do load_train, depois o teste).

Uso:
    python members_extra.py --data train.xlsx --test test.xlsx --folds_from "runs_v2/final_large_*"
    python members_extra.py --data train.xlsx --members knn_tfidf nb_tfidf hgb_stats        # sem GPU
"""
import argparse
import glob
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler

from data_utils import LABELS, load_train, clean_text
from train_v2 import make_group_folds, metrics_from_probs

ALL = ["knn_tfidf", "nb_tfidf", "hgb_stats", "knn_emb", "lr_emb", "knn_grid"]
K3 = len(LABELS)


# ───────────────────────────── kNN ─────────────────────────────

def knn_probs(S, ytr, k, power=None, temp=None, eps=0.1, inv_dist=False):
    """S = similaridades (n_consulta x n_treino). Votos dos k mais parecidos, ponderados por sim**power
    (esparso/TF-IDF) ou exp((sim - sim_max)/temp) (embeddings). Mistura com a frequência das classes."""
    k = min(k, S.shape[1])
    idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
    sims = np.take_along_axis(S, idx, 1)
    if inv_dist:                                   # weights='distance' do sklearn: 1/(1-cos)
        w = 1.0 / np.clip(1.0 - sims, 1e-6, None)
    elif temp:
        w = np.exp((sims - sims.max(1, keepdims=True)) / temp)
    else:
        w = np.clip(sims, 0, None) ** power
    onehot = np.eye(K3, dtype=np.float32)[ytr[idx]]
    p = (w[..., None] * onehot).sum(1) / np.clip(w.sum(1, keepdims=True), 1e-9, None)
    prior = np.bincount(ytr, minlength=K3) / len(ytr)
    return ((1 - eps) * p + eps * prior).astype(np.float32)


def knn_chunks(Xq, Xtr, ytr, k, dense, **kw):
    out = []
    for i in range(0, Xq.shape[0], 1000):
        S = Xq[i:i + 1000] @ Xtr.T
        S = np.asarray(S) if dense else S.toarray()
        out.append(knn_probs(S, ytr, k, **kw))
    return np.vstack(out)


# ───────────────────────────── estatísticas do texto ─────────────────────────────

_TERMS = ["lei", "art", "decreto", "portaria", "conforme", "informamos", "anexo", "sic", "protocolo", "http",
          "gov.br", "não", "disponível", "recurso", "prazo", "solicitação", "pedido", "dados", "ouvidoria",
          "esclarecemos", "cumprimento", "atenciosamente", "prezad", "senhor", "obrigad"]


def stats_features(texts):
    rows = []
    for t in texts:
        low = t.lower()
        words = re.findall(r"\w+", low)
        sents = [s for s in re.split(r"(?<=[.!?;:])\s+", t) if s.strip()]
        nw = max(len(words), 1)
        r = {"n_chars": len(t), "n_words": len(words), "n_sents": len(sents),
             "words_per_sent": len(words) / max(len(sents), 1),
             "avg_word_len": sum(map(len, words)) / nw,
             "uniq_ratio": len(set(words)) / nw,
             "digit_share": sum(c.isdigit() for c in t) / max(len(t), 1),
             "upper_share": sum(c.isupper() for c in t) / max(len(t), 1),
             "punct_share": sum(c in ",.;:!?()" for c in t) / max(len(t), 1),
             "n_urls": len(re.findall(r"https?://|www\.", low)),
             "n_emails": len(re.findall(r"\S+@\S+", low)),
             "n_long_words": sum(len(w) >= 12 for w in words),
             "n_parens": t.count("("), "n_quotes": t.count('"'), "n_colons": t.count(":"),
             "first_upper": float(t[:1].isupper()), "ends_period": float(t.rstrip()[-1:] in ".!?")}
        for term in _TERMS:
            r["t_" + term] = low.count(term)
        rows.append(r)
    return pd.DataFrame(rows)


# ───────────────────────────── embeddings ─────────────────────────────

def get_embeddings(texts, args):
    if args.emb_npy:
        E = np.load(args.emb_npy).astype(np.float32)
        assert len(E) == len(texts), f"--emb_npy tem {len(E)} linhas, esperado {len(texts)} (treino + teste)"
        return E
    key = hashlib.md5((args.emb_model + args.emb_prefix + str(args.emb_max_len) + "\n".join(texts)).encode()).hexdigest()[:10]
    cache = Path(f"emb_{args.emb_model.split('/')[-1]}_{key}.npy")
    if cache.exists():
        print(f"[info] embeddings do cache: {cache}")
        return np.load(cache)
    from sentence_transformers import SentenceTransformer
    import torch
    m = SentenceTransformer(args.emb_model, device="cuda" if torch.cuda.is_available() else "cpu")
    m.max_seq_length = args.emb_max_len
    print(f"[info] calculando embeddings com {args.emb_model} ({len(texts)} textos, max {args.emb_max_len} tokens)...")
    E = m.encode([args.emb_prefix + t for t in texts], batch_size=args.emb_bs, normalize_embeddings=True,
                 show_progress_bar=True).astype(np.float32)
    np.save(cache, E)
    return E


# ───────────────────────────── execução dos membros ─────────────────────────────

def run_member(name, df, fold, K, test_texts, args, E=None):
    y = df["label_id"].values
    texts = df["text"].values
    oof = np.zeros((len(df), K3), dtype=np.float32)
    tests = []
    Et = None
    if E is not None:
        E = E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-9, None)     # cosseno = produto interno
        E, Et = E[:len(df)], (E[len(df):] if test_texts is not None else None)
    t0 = time.time()
    if name == "hgb_stats":                      # estatísticas não dependem do fold: calcula 1x
        S_cols = stats_features(texts[:1]).columns
        S_all = stats_features(texts)[S_cols].values
        S_te = stats_features(test_texts)[S_cols].values if test_texts is not None else None
    for k in range(K):
        tr, va = np.where(fold != k)[0], np.where(fold == k)[0]
        if name in ("knn_tfidf", "nb_tfidf", "hgb_stats", "knn_grid"):
            if name == "knn_grid":     # melhor combinação do grid do usuário (acc. média 43,019%)
                vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), min_df=1, sublinear_tf=True)
            else:
                vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
            Xtr = vec.fit_transform(texts[tr])
            Xva = vec.transform(texts[va])
            Xte = vec.transform(test_texts) if test_texts is not None else None
            if name == "knn_tfidf":
                f = lambda Xq: knn_chunks(Xq, Xtr, y[tr], args.knn_k, dense=False, power=args.knn_power)
            elif name == "knn_grid":
                f = lambda Xq: knn_chunks(Xq, Xtr, y[tr], args.grid_k, dense=False, inv_dist=True)
            elif name == "nb_tfidf":
                clf = MultinomialNB(alpha=args.nb_alpha).fit(Xtr, y[tr])
                f = clf.predict_proba
            else:
                svd = TruncatedSVD(64, random_state=0).fit(Xtr)
                clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                     l2_regularization=1.0, random_state=0)
                clf.fit(np.hstack([S_all[tr], svd.transform(Xtr)]), y[tr])
                P_va = clf.predict_proba(np.hstack([S_all[va], svd.transform(Xva)]))
                P_te = (clf.predict_proba(np.hstack([S_te, svd.transform(Xte)])) if Xte is not None else None)
            if name != "hgb_stats":
                P_va = f(Xva)
                P_te = f(Xte) if Xte is not None else None
        else:   # knn_emb / lr_emb
            if name == "knn_emb":
                f = lambda Eq: knn_chunks(Eq, E[tr], y[tr], args.knn_k, dense=True, temp=args.knn_temp)
            else:
                sc = StandardScaler().fit(E[tr])
                clf = LogisticRegression(C=args.lr_C, max_iter=2000).fit(sc.transform(E[tr]), y[tr])
                f = lambda Eq: clf.predict_proba(sc.transform(Eq))
            P_va = f(E[va])
            P_te = f(Et) if Et is not None else None
        oof[va] = P_va
        if P_te is not None:
            tests.append(np.asarray(P_te, dtype=np.float32))
    m = metrics_from_probs(y, oof)
    print(f"  {name:10s} acc={m['accuracy']:.4f} macro-F1={m['macro_f1']:.4f} log-loss={m['log_loss']:.4f} "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    return oof, (np.mean(tests, axis=0) if tests else None), m, (time.time() - t0) / 60


def save_member(name, oof, test, m, minutes, df, fold, args):
    d = Path(args.out_dir) / f"X_{name}_{int(time.time())}"
    d.mkdir(parents=True, exist_ok=True)
    o = pd.DataFrame(oof, columns=[f"p_{l}" for l in LABELS])
    o.insert(0, "fold", fold)
    o.insert(0, "clarity", df["clarity"].values)
    o.to_csv(d / "oof_probs.csv", index_label="row")
    if test is not None:
        np.save(d / "test_probs_mean.npy", test)
    json.dump({"overall_oof": m, "minutes": minutes, "rows_covered": len(df)}, open(d / "summary.json", "w"), indent=2)
    json.dump({k: v for k, v in vars(args).items()}, open(d / "config.json", "w"), indent=2)
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="train.xlsx")
    p.add_argument("--test", default=None)
    p.add_argument("--members", nargs="*", default=ALL, choices=ALL)
    p.add_argument("--kfold", type=int, default=5)
    p.add_argument("--folds_from", default=None, help="padrão glob de uma rodada anterior (usa os folds dela)")
    p.add_argument("--out_dir", default="runs_v2")
    p.add_argument("--knn_k", type=int, default=50)
    p.add_argument("--knn_power", type=float, default=2.0, help="peso = sim**power (TF-IDF)")
    p.add_argument("--grid_k", type=int, default=15, help="k do membro knn_grid (config do grid: cosseno, weights=distance)")
    p.add_argument("--knn_temp", type=float, default=0.05, help="temperatura dos pesos exp(sim/T) (embeddings)")
    p.add_argument("--nb_alpha", type=float, default=0.3)
    p.add_argument("--lr_C", type=float, default=0.05)
    p.add_argument("--emb_model", default="intfloat/multilingual-e5-base")
    p.add_argument("--emb_prefix", default="query: ", help="o e5 pede 'query: '; deixe vazio para outros modelos")
    p.add_argument("--emb_max_len", type=int, default=512)
    p.add_argument("--emb_bs", type=int, default=32)
    p.add_argument("--emb_npy", default=None)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    df = load_train(args.data)
    if args.debug:
        df = df.sample(min(1500, len(df)), random_state=0).reset_index(drop=True)
    if args.folds_from:
        src = sorted(glob.glob(args.folds_from))[-1]
        fold = pd.read_csv(Path(src) / "oof_probs.csv")["fold"].values
        assert len(fold) == len(df), f"{src}: {len(fold)} linhas, esperado {len(df)} (mesmo --data?)"
        print(f"[info] folds lidos de {src}")
    else:
        fold = make_group_folds(df, args.kfold)
    K = int(fold.max()) + 1
    test_texts = None
    if args.test:
        test_texts = pd.read_excel(args.test)["resp_text"].apply(clean_text).tolist()

    E = None
    if any(m.endswith("_emb") for m in args.members):
        try:
            E = get_embeddings(list(df["text"]) + (test_texts or []), args)
        except Exception as e:
            print(f"[aviso] embeddings indisponíveis ({type(e).__name__}: {str(e)[:100]}); "
                  f"pulando knn_emb e lr_emb.")
            args.members = [m for m in args.members if not m.endswith("_emb")]

    print(f"\n{len(df)} linhas | {K} folds | membros: {', '.join(args.members)}")
    for name in args.members:
        oof, test, m, mins = run_member(name, df, fold, K, test_texts, args, E if name.endswith("_emb") else None)
        d = save_member(name, oof, test, m, mins, df, fold, args)
        print(f"    salvo em {d}")


if __name__ == "__main__":
    main()
