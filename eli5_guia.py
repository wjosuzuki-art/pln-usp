"""
eli5_guia.py -- insights do eli5 (TF-IDF + Reg. Log.) para alimentar o LLM, SEM olhar o teste.

  guia   : usa só o TREINO. Ajusta o modelo em 5 folds (GroupKFold por texto) e mantém apenas termos ESTÁVEIS
           (mesma classe dominante em >= --min_folds folds) e com suporte (>= --min_df textos). Para cada termo mostra a
           distribuição real de rótulos no treino. Termos que a Reg. Log. "ama" mas cuja distribuição é quase plana
           (ex.: 'solicitação') vão para a lista de armadilhas. Gera guia.md (para colar no prompt do LLM).
  pistas : para os suspeitos de um prep, gera pistas.csv com os termos que mais pesam, via eli5.explain_prediction_df.

Uso:
  python eli5_guia.py guia   --train train.xlsx --out guia.md
  python eli5_guia.py pistas --train train.xlsx --suspeitos llm_t1/suspeitos.csv --out llm_t1/pistas.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from data_utils import LABELS, clean_text, load_train  # noqa: E402


def novo_vec():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=60000)


def ajusta(textos, y):
    vec = novo_vec(); X = vec.fit_transform(textos)
    clf = LogisticRegression(C=4.0, max_iter=2000).fit(X, y)
    return vec, clf


def cmd_guia(a):
    tr = load_train(a.train)
    t, y = tr["text"].values, tr["label_id"].values
    grupos = pd.factorize(t)[0]
    # 1) pesos por fold -> classe dominante de cada termo e estabilidade
    dom, vocs = [], []
    for f, (i, _) in enumerate(GroupKFold(5).split(t, y, grupos)):
        vec, clf = ajusta(t[i], y[i])
        voc = np.array(vec.get_feature_names_out())
        vocs.append(pd.Series(clf.coef_.argmax(0), index=voc)); 
    D = pd.concat(vocs, axis=1)
    # 2) modelo final + eli5 (pesos globais)
    vec, clf = ajusta(t, y)
    voc = np.array(vec.get_feature_names_out())
    try:
        import eli5
        df = eli5.explain_weights_df(clf, vec=vec, top=400, target_names=LABELS)
        w = df[df.feature != "<BIAS>"].rename(columns={"target": "classe", "feature": "termo", "weight": "peso"})
    except Exception as e:  # eli5 indisponível: usa os mesmos coeficientes
        print("eli5 falhou (", e, ") -> usando coef_ direto")
        rows = [(LABELS[c], voc[j], clf.coef_[c, j]) for c in range(3) for j in np.argsort(-clf.coef_[c])[:400]]
        w = pd.DataFrame(rows, columns=["classe", "termo", "peso"])
    # 3) distribuição real de rótulos por termo (documentos únicos que contêm o termo)
    X = vec.transform(t) > 0
    jmap = {v: k for k, v in enumerate(voc)}
    linhas, armadilhas = [], []
    for _, r in w.iterrows():
        j = jmap.get(r.termo)
        if j is None: continue
        c = LABELS.index(r.classe)
        n = int(X[:, j].sum())
        if n < a.min_df: continue
        est = int((D.loc[r.termo].dropna() == c).sum()) if r.termo in D.index else 0
        dist = np.bincount(y[X[:, j].nonzero()[0]], minlength=3) / n
        item = (r.classe, r.termo, round(float(r.peso), 2), n, est, dist)
        if est >= a.min_folds and dist[c] >= a.min_pureza: linhas.append(item)
        elif est >= a.min_folds: armadilhas.append(item)
    out = ["# Guia de pistas lexicais (derivado SÓ do treino; 5 folds; termos estáveis)", "",
           "Use como evidência fraca, nunca como regra. Entre [colchetes]: % de c1/c234/c5 entre os textos de treino que contêm o termo.", ""]
    for c, l in enumerate(LABELS):
        sel = sorted([x for x in linhas if x[0] == l], key=lambda x: -x[5][c])[:a.top]
        out += [f"## Termos que empurram para {l}"] + [f"- '{x[1]}' (n={x[3]}) [{x[5][0]*100:.0f}/{x[5][1]*100:.0f}/{x[5][2]*100:.0f}]" for x in sel] + [""]
    arm = sorted(armadilhas, key=lambda x: -x[3])[:a.top]
    out += ["## Armadilhas: a Reg. Log. dá peso, mas a distribuição real é quase plana (NÃO use como pista)"]
    out += [f"- '{x[1]}' (n={x[3]}) [{x[5][0]*100:.0f}/{x[5][1]*100:.0f}/{x[5][2]*100:.0f}]" for x in arm]
    Path(a.out).write_text("\n".join(out), encoding="utf-8")
    print(f"{len(linhas)} termos úteis, {len(armadilhas)} armadilhas -> {a.out}")


def cmd_pistas(a):
    import eli5
    tr = load_train(a.train)
    vec, clf = ajusta(tr["text"].values, tr["label_id"].values)
    sus = pd.read_csv(a.suspeitos)
    res = []
    for _, r in sus.iterrows():
        d = eli5.explain_prediction_df(clf, r["texto"], vec=vec, target_names=LABELS, top=(4, 0))
        d = d[d.feature != "<BIAS>"]
        res.append({"linha": r["linha"], "pistas": "; ".join(f"{f}→{tg}" for tg, f in zip(d.target, d.feature))})
    pd.DataFrame(res).to_csv(a.out, index=False)
    print("salvo:", a.out)


p = argparse.ArgumentParser(); sp = p.add_subparsers(dest="cmd", required=True)
g = sp.add_parser("guia"); g.add_argument("--train", default="train.xlsx"); g.add_argument("--out", default="guia.md")
g.add_argument("--min_df", type=int, default=30); g.add_argument("--min_folds", type=int, default=4)
g.add_argument("--min_pureza", type=float, default=0.42); g.add_argument("--top", type=int, default=12)
q = sp.add_parser("pistas"); q.add_argument("--train", default="train.xlsx"); q.add_argument("--suspeitos", required=True); q.add_argument("--out", default="pistas.csv")
a = p.parse_args(); {"guia": cmd_guia, "pistas": cmd_pistas}[a.cmd](a)
