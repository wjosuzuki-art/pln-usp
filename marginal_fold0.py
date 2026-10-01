"""
marginal_fold0.py -- quanto cada modelo candidato melhora o ensemble ATUAL?

Serve para decidir, ANTES de gastar horas de GPU nos 5 folds, quais modelos novos valem
a pena. Usa só previsões que já existem: as rodadas completas (5 folds) do ensemble atual
e as rodadas do grid (que só cobrem o fold 0). Compara, nas linhas do fold 0:

    ensemble atual  (média simples: modelos-base [+ TF-IDF])
    ensemble atual + candidato

e diz se a diferença é real com bootstrap pareado por TEXTO (mesma lógica do compare_oof.py).
Regra prática: só o intervalo de confiança que NÃO inclui zero é evidência; com ~4.000
linhas, a acurácia sozinha só detecta ganhos acima de ~1 ponto, então olhe a log-loss.

Uso:
    python marginal_fold0.py --data train.xlsx --add_tfidf \
        --base runs_v2/final_large_1790601626 runs_v2/final_legal_1790612371 \
        --cands g_norb_lr3e-5 g_norb_lr5e-5 g_alb_lora16_lr1e-4 g_alb_lora16_lr2e-4 g_norb_lr2e-5

--cands aceita o TAG do grid (procura em runs_v2/) ou o caminho da pasta.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from data_utils import LABELS, load_train


def find_run_dir(spec, runs_dir):
    p = Path(spec)
    if (p / "oof_probs.csv").exists():
        return p
    pat = re.compile(rf"^{re.escape(spec)}_\d+$")
    runs = [d for d in Path(runs_dir).glob(f"{spec}_*") if pat.match(d.name) and (d / "oof_probs.csv").exists()]
    assert runs, f"não achei rodada '{spec}' em {runs_dir}/ (nem como pasta)"
    return max(runs, key=lambda d: d.stat().st_mtime)


def load_oof(run_dir):
    o = pd.read_csv(Path(run_dir) / "oof_probs.csv")
    return o[[f"p_{l}" for l in LABELS]].values.astype(np.float64), o["fold"].values, o["clarity"].values


def per_row(P, y):
    P = P / P.sum(1, keepdims=True)
    return (P.argmax(1) == y).astype(float), -np.log(np.clip(P[np.arange(len(y)), y], 1e-7, 1))


def cluster_bootstrap(d_acc, d_ll, text, n_boot=2000, seed=0):
    """Bootstrap pareado, sorteando TEXTOS (cópias idênticas não são amostras independentes)."""
    _, g = np.unique(text, return_inverse=True)
    G = g.max() + 1
    n_g = np.bincount(g, minlength=G).astype(float)
    sa = np.bincount(g, weights=d_acc, minlength=G)
    sl = np.bincount(g, weights=d_ll, minlength=G)
    idx = np.random.default_rng(seed).integers(0, G, size=(n_boot, G))
    den = n_g[idx].sum(1)
    return np.percentile(sa[idx].sum(1) / den, [2.5, 97.5]), np.percentile(sl[idx].sum(1) / den, [2.5, 97.5])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="train.xlsx")
    p.add_argument("--base", nargs="+", required=True, help="Pastas das rodadas COMPLETAS do ensemble atual.")
    p.add_argument("--cands", nargs="+", required=True, help="Tags do grid (ou pastas) a avaliar.")
    p.add_argument("--add_tfidf", action="store_true", help="Inclui o TF-IDF+LogReg no ensemble-base.")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--runs_dir", default="runs_v2")
    args = p.parse_args()

    df = load_train(args.data)
    y_all, text_all = df["label_id"].values, df["text"].values

    base, fold = [], None
    for b in args.base:
        P, f, c = load_oof(b)
        assert len(P) == len(df) and (c == df["clarity"].values).all(), f"{b}: não bate com o train.xlsx"
        assert not np.isnan(P).any(), f"{b}: OOF incompleto (só rodadas de 5 folds completos entram em --base)"
        fold = f if fold is None else fold
        assert (f == fold).all(), f"{b}: esquema de folds diferente do primeiro --base"
        base.append(P)
    k = args.fold
    rows = fold == k

    if args.add_tfidf:
        print(f"[info] ajustando o TF-IDF+LogReg nos folds != {k} (uns 30-60 s)...")
        vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        clf = LogisticRegression(max_iter=1000).fit(vec.fit_transform(text_all[~rows]), y_all[~rows])
        T = np.zeros((len(df), 3))
        T[rows] = clf.predict_proba(vec.transform(text_all[rows]))
        base.append(T)

    cands = {}
    for spec in args.cands:
        d = find_run_dir(spec, args.runs_dir)
        P, f, c = load_oof(d)
        assert (c == df["clarity"].values).all() and (f == fold).all(), \
            f"{d}: folds diferentes dos do --base (o grid usou --kfold 5?)"
        if np.isnan(P[rows]).any():
            print(f"[aviso] {d.name}: sem previsões completas no fold {k} -- ignorado")
            continue
        cands[spec] = P

    y, text = y_all[rows], text_all[rows]
    B = sum(P[rows] for P in base) / len(base)
    acc_b, ll_b = per_row(B, y)
    print(f"\nfold {k}: {rows.sum()} linhas | ensemble atual = média simples de {len(base)} membros "
          f"({len(args.base)} BERT{' + TF-IDF' if args.add_tfidf else ''})")
    print(f"  ensemble atual:             acc={acc_b.mean():.4f}  log-loss={ll_b.mean():.4f}\n")
    print(f"{'candidato':26s} {'sozinho':>8s} {'concorda':>9s} | {'ens.+cand acc':>13s} {'Δacc [IC95%]':>26s} | {'Δlog-loss [IC95%]':>28s}  veredito")

    for name, P in cands.items():
        Pc = P[rows]
        alone = (Pc.argmax(1) == y).mean()
        agree = (Pc.argmax(1) == B.argmax(1)).mean()
        E = (sum(Q[rows] for Q in base) + Pc) / (len(base) + 1)
        acc_e, ll_e = per_row(E, y)
        (alo, ahi), (llo, lhi) = cluster_bootstrap(acc_e - acc_b, ll_e - ll_b, text)
        da, dl = acc_e.mean() - acc_b.mean(), ll_e.mean() - ll_b.mean()
        if lhi < 0:
            verdict = "AJUDA (log-loss)"
        elif alo > 0:
            verdict = "AJUDA (acurácia)"
        elif llo > 0 or ahi < 0:
            verdict = "ATRAPALHA"
        elif da > 0 or dl < 0:
            verdict = "sinal fraco"
        else:
            verdict = "não ajuda"
        print(f"{name:26s} {alone:8.4f} {agree:9.1%} | {acc_e.mean():13.4f} {da:+8.4f} [{alo:+.4f},{ahi:+.4f}] | "
              f"{dl:+8.4f} [{llo:+.4f},{lhi:+.4f}]  {verdict}")

    print("\n'concorda' = % das linhas em que o candidato prevê a mesma classe que o ensemble atual "
          "(quanto MENOR, mais diferente ele é, e mais chance de complementar).")
    print("Com ~4.000 linhas, 'sinal fraco' e 'não ajuda' NÃO significam que o modelo é inútil: significam "
          "que este fold não consegue distinguir. Só 'AJUDA' é evidência.")


if __name__ == "__main__":
    main()