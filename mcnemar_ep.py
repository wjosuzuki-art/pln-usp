"""
mcnemar_ep.py -- tabela EP1 x EP2 no teste 1 e teste de McNemar (exigido no item 8 do relatório do EP2).

Uso:
    python mcnemar_ep.py --labels test1-rotulado-para-teste-estatistico.xlsx --ep1 <planilha entregue no EP1>.xlsx --ep2 ens_core_t1.xlsx

Cada planilha precisa ter as colunas resp_text e clarity (a ordem das linhas é conferida pelo texto).
Imprime: acurácia de cada um, a tabela de acertos/erros em pares, a estatística qui-quadrado de McNemar (com correção de
continuidade), o p-valor exato (binomial bilateral) e o texto pronto para colar no relatório.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True)
    p.add_argument("--ep1", required=True)
    p.add_argument("--ep2", required=True)
    a = p.parse_args()
    lab, e1, e2 = (pd.read_excel(f) for f in (a.labels, a.ep1, a.ep2))
    assert len(lab) == len(e1) == len(e2), f"linhas diferentes: {len(lab)} / {len(e1)} / {len(e2)}"
    for nome, d in (("ep1", e1), ("ep2", e2)):
        if "resp_text" in d and not (d["resp_text"].astype(str).values == lab["resp_text"].astype(str).values).all():
            raise SystemExit(f"{nome}: textos em ordem diferente dos rótulos do test1")
    y = lab["clarity"].values
    c1, c2 = e1["clarity"].values == y, e2["clarity"].values == y
    n = len(y)
    b = int((c1 & ~c2).sum())     # EP1 acerta, EP2 erra
    c = int((~c1 & c2).sum())     # EP1 erra, EP2 acerta
    both, none = int((c1 & c2).sum()), int((~c1 & ~c2).sum())
    chi = (abs(b - c) - 1) ** 2 / (b + c) if b + c else 0.0
    p_chi = float(chi2.sf(chi, 1)) if b + c else 1.0
    p_ex = float(binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0
    print(f"n = {n}")
    print(f"EP1: {c1.mean() * 100:.2f}%  ({c1.sum()} acertos)")
    print(f"EP2: {c2.mean() * 100:.2f}%  ({c2.sum()} acertos)   diferença {100 * (c2.mean() - c1.mean()):+.2f} ponto")
    print("\nTabela de pares:           EP2 certo   EP2 errado")
    print(f"  EP1 certo                {both:9d}   {b:10d}")
    print(f"  EP1 errado               {c:9d}   {none:10d}")
    print(f"\nMcNemar: b={b} (só EP1 acerta), c={c} (só EP2 acerta)")
    print(f"  qui-quadrado (1 gl, correção de continuidade) = {chi:.3f}, p = {p_chi:.4f}")
    print(f"  p exato (binomial bilateral)                  = {p_ex:.4f}")
    sig = p_ex < 0.05 and c > b
    print("\n--- texto para o relatório ---")
    print(f"| Estratégia | Teste | Acurácia |\n| Modelo EP1 | teste 1 | {c1.mean() * 100:.1f}% |\n| Modelo EP2 | teste 1 | {c2.mean() * 100:.1f}% |")
    if sig:
        print(f"A vantagem do modelo 2 é estatisticamente significativa de acordo com um teste de McNemar (χ² = {chi:.2f}, p = {p_ex:.4f}, exato).")
    else:
        print(f"A diferença entre os modelos NÃO é estatisticamente significativa de acordo com um teste de McNemar "
              f"(χ² = {chi:.2f}, p = {p_ex:.4f}, exato; {c} acertos exclusivos do EP2 contra {b} do EP1).")


if __name__ == "__main__":
    main()
