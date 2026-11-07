"""
prior_shift.py -- corrige o deslocamento de prioris entre treino e teste (Saerens et al., 2002), SEM usar rótulos do teste.
O modelo aprende com as frequências do treino; se o teste tem outra proporção de classes (aqui: ~33/33/33), as
probabilidades ficam enviesadas. Ajusta pesos w por classe (EM) para que a média das probabilidades no teste vire a
priori alvo (uniforme por padrão) e reaplica o argmax.
Uso:
  python prior_shift.py --test test1.xlsx --probs ens9_t1_probs.npy --out ens9_t1_ps.xlsx --labels test1-rotulado-para-teste-estatistico.xlsx
  python prior_shift.py --test test2.xlsx --probs <probs test2> --out test2_ps.xlsx
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from data_utils import LABELS

p = argparse.ArgumentParser()
p.add_argument("--test", required=True); p.add_argument("--probs", required=True); p.add_argument("--out", required=True)
p.add_argument("--labels", default=None); p.add_argument("--alvo", type=float, nargs=3, default=[1/3, 1/3, 1/3])
a = p.parse_args()
P = np.load(a.probs); te = pd.read_excel(a.test); assert len(P) == len(te)
alvo = np.array(a.alvo) / sum(a.alvo)
w = np.ones(3)
for _ in range(500):
    Q = P * w; Q /= Q.sum(1, keepdims=True)
    w *= (alvo / Q.mean(0)) ** 0.5
Q = P * w; Q /= Q.sum(1, keepdims=True)
b, n = P.argmax(1), Q.argmax(1)
print("pesos por classe:", np.round(w / w.sum() * 3, 3))
print("previsões antes :", np.bincount(b, minlength=3), "| depois:", np.bincount(n, minlength=3), "| mudaram:", int((b != n).sum()))
if a.labels:
    y = pd.read_excel(a.labels)["clarity"].map({l: i for i, l in enumerate(LABELS)}).values
    ab, an = b == y, n == y
    print(f"acurácia: antes {ab.mean()*100:.2f}%  depois {an.mean()*100:.2f}%  ({int(an.sum()-ab.sum()):+d} acertos)")
    print(f"das {int((b!=n).sum())} trocas: {int((~ab&an).sum())} errado->certo | {int((ab&~an).sum())} certo->errado")
te["clarity"] = [LABELS[i] for i in n]; te.to_excel(a.out, index=False); print("salvo:", a.out)
