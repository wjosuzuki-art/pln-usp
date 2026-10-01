"""
compare_oof.py -- compara duas rodadas (ex.: BERTimbau-base com e sem TAPT) usando as
previsões out-of-fold de cada uma, e diz se a diferença é real ou pode ser sorte.

Como: bootstrap PAREADO por TEXTO -- sorteia textos (com reposição), soma acertos e
log-loss das duas rodadas nos mesmos textos, e repete 2000 vezes. Agrupar por texto é
importante porque cópias idênticas (12% do corpus) não são amostras independentes.
Regra prática: se o intervalo de confiança de 95% da diferença inclui zero, não dá para
afirmar que uma é melhor que a outra.

Uso:
    python compare_oof.py runs_v2/base_normal_XXXX runs_v2/base_tapt_XXXX --data train.xlsx
    (a diferença é sempre  SEGUNDA - PRIMEIRA)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data_utils import LABELS, load_train


def load_oof(path, n_rows):
    o = pd.read_csv(Path(path) / "oof_probs.csv")
    assert len(o) == n_rows, f"{path}: OOF com {len(o)} linhas, esperado {n_rows} (mesmo train.xlsx?)"
    return o[[f"p_{l}" for l in LABELS]].values.astype(np.float64), o["clarity"].values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.add_argument("--data", default="train.xlsx")
    p.add_argument("--n_boot", type=int, default=2000)
    args = p.parse_args()

    df = load_train(args.data)
    y = df["label_id"].values
    Pa, ca = load_oof(args.run_a, len(df))
    Pb, cb = load_oof(args.run_b, len(df))
    assert (ca == df["clarity"].values).all() and (cb == df["clarity"].values).all(), "rótulos não batem com o train.xlsx"
    ok = ~(np.isnan(Pa).any(1) | np.isnan(Pb).any(1))
    if not ok.all():
        print(f"[aviso] só {ok.sum()} de {len(ok)} linhas têm previsão nas duas rodadas (comparando nessas).")
    Pa, Pb, y, text = Pa[ok], Pb[ok], y[ok], df["text"].values[ok]

    def per_row(P):
        P = P / P.sum(1, keepdims=True)
        return (P.argmax(1) == y).astype(float), -np.log(np.clip(P[np.arange(len(y)), y], 1e-7, 1))
    acc_a, ll_a = per_row(Pa)
    acc_b, ll_b = per_row(Pb)
    print(f"A = {args.run_a}\n   acc={acc_a.mean():.4f}  log-loss={ll_a.mean():.4f}")
    print(f"B = {args.run_b}\n   acc={acc_b.mean():.4f}  log-loss={ll_b.mean():.4f}")

    only_b = int(((acc_b == 1) & (acc_a == 0)).sum())
    only_a = int(((acc_a == 1) & (acc_b == 0)).sum())
    print(f"\nlinhas em que só A acerta: {only_a} | só B acerta: {only_b} | "
          f"concordam nas previsões: {(Pa.argmax(1) == Pb.argmax(1)).mean():.1%}")

    # bootstrap pareado, agrupado por texto
    _, g = np.unique(text, return_inverse=True)
    G = g.max() + 1
    n_g = np.bincount(g, minlength=G).astype(float)
    d_acc = np.bincount(g, weights=acc_b - acc_a, minlength=G)
    d_ll = np.bincount(g, weights=ll_b - ll_a, minlength=G)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, G, size=(args.n_boot, G))
    den = n_g[idx].sum(1)
    boot_acc = d_acc[idx].sum(1) / den
    boot_ll = d_ll[idx].sum(1) / den

    def show(name, obs, boot, better_if_positive):
        lo, hi = np.percentile(boot, [2.5, 97.5])
        verdict = "diferença não distinguível de zero"
        if lo > 0 or hi < 0:
            good = (obs > 0) == better_if_positive
            verdict = f"B é {'MELHOR' if good else 'PIOR'} que A (IC não inclui zero)"
        print(f"  {name:9s} B - A = {obs:+.4f}  IC95% [{lo:+.4f}, {hi:+.4f}]  -> {verdict}")

    print(f"\nBootstrap pareado por texto ({G} textos, {args.n_boot} reamostragens):")
    show("acurácia", acc_b.mean() - acc_a.mean(), boot_acc, True)
    show("log-loss", ll_b.mean() - ll_a.mean(), boot_ll, False)
    print("(log-loss costuma ser bem menos ruidosa que a acurácia: use-a para julgar diferenças pequenas)")


if __name__ == "__main__":
    main()