"""
score_test1.py -- acurácia REAL no test1 (agora rotulado) de tudo que já foi gerado:
  * previsões de teste salvas nas pastas de execução (test_probs_mean.npy, test_probs_full.npy,
    test_probs_dupblend.npy);
  * planilhas de submissão (.xlsx com coluna clarity).
Não treina nada; leva segundos.

Uso:
    python score_test1.py --labels test1-rotulado-para-teste-estatistico.xlsx
    python score_test1.py --labels test1-rotulado-para-teste-estatistico.xlsx --subs submission_*.xlsx ens_*/*.xlsx
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ["c1", "c234", "c5"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True)
    p.add_argument("--runs", nargs="*", default=None, help="pastas (padrão: todas em runs_v2/)")
    p.add_argument("--subs", nargs="*", default=[], help="planilhas de submissão .xlsx")
    a = p.parse_args()

    lab = pd.read_excel(a.labels)
    y = lab["clarity"].map({l: i for i, l in enumerate(LABELS)}).values
    n = len(y)
    print(f"test1: {n} linhas | " + " ".join(f"{l} {np.mean(y == i):.1%}" for i, l in enumerate(LABELS)))
    print(f"erro-padrão de uma acurácia ~50% com {n} textos: ~{np.sqrt(.25 / n) * 100:.1f} ponto\n")

    rows = []
    runs = a.runs if a.runs is not None else sorted(glob.glob("runs_v2/*"))
    for r in runs:
        for f in ("test_probs_mean.npy", "test_probs_full.npy", "test_probs_dupblend.npy"):
            fp = Path(r) / f
            if fp.exists():
                P = np.load(fp)
                if len(P) != n:
                    continue
                ll = -np.log(np.clip(P[np.arange(n), y], 1e-7, 1)).mean()
                rows.append((f"{Path(r).name}/{f.replace('test_probs_', '').replace('.npy', '')}",
                             (P.argmax(1) == y).mean(), ll))
    for s in a.subs:
        for f in glob.glob(s):
            d = pd.read_excel(f)
            if "clarity" in d and len(d) == n:
                if "resp_text" in d and not (d["resp_text"].astype(str).values ==
                                             lab["resp_text"].astype(str).values).all():
                    print(f"[aviso] {f}: textos em ordem diferente do test1 -- ignorado")
                    continue
                rows.append((f, (d["clarity"].values == lab["clarity"].values).mean(), float("nan")))

    for name, acc, ll in sorted(rows, key=lambda t: -t[1]):
        print(f"{acc * 100:6.2f}%  log-loss {ll:.4f}  {name}" if ll == ll else f"{acc * 100:6.2f}%                   {name}")


if __name__ == "__main__":
    main()
