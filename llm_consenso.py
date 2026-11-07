"""
llm_consenso.py -- correção dos casos mais incertos do ensemble por consenso de LLMs (3 etapas).

  1) prep   : escolhe os N casos de menor confiança do ensemble ENTRE OS QUE NÃO SÃO CÓPIA EXATA de um texto do treino
              (esses já são tratados pelo dupblend) e monta lotes de texto para colar nos LLMs, cada caso com os textos
              mais parecidos DO TREINO (cosseno TF-IDF) e a contagem de rótulos de cada um.
  2) (você) : cola cada lote em cada LLM e salva a resposta como CSV com colunas  linha,rotulo   (rotulo em c1/c234/c5).
  3) apply  : troca o rótulo do ensemble só quando >= --min_votos LLMs concordam entre si num rótulo DIFERENTE do
              modelo; com --labels (test1) mede ganho/perda contra os rótulos verdadeiros.

Uso:
  python llm_consenso.py prep  --train train.xlsx --test test.xlsx --probs ens_core_t1_probs.npy --n 100 --out_dir llm_t1
  python llm_consenso.py apply --test test.xlsx --probs ens_core_t1_probs.npy --suspeitos llm_t1/suspeitos.csv \
         --respostas llm_t1/a.csv llm_t1/b.csv llm_t1/c.csv --labels test1-rotulado-para-teste-estatistico.xlsx --out ens_llm_t1.xlsx
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data_utils import LABELS, clean_text, load_train  # noqa: E402


def suspeitos(P, n, permitido=None):
    srt = np.sort(P, axis=1)
    margem = srt[:, -1] - srt[:, -2]
    ordem = np.argsort(margem)
    if permitido is not None:
        ordem = np.array([i for i in ordem if permitido[i]])
    return ordem[:n], margem


def cmd_prep(a):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tr = load_train(a.train)
    te = pd.read_excel(a.test)
    tt = te["resp_text"].apply(clean_text)
    P = np.load(a.probs)
    assert len(P) == len(te), f"probs tem {len(P)} linhas e o teste {len(te)}"
    dup = tt.isin(set(tr["text"])).values
    print(f"{dup.sum()} de {len(te)} textos do teste são cópias exatas de textos do treino (tratados pelo dupblend, fora dos suspeitos)")
    idx, margem = suspeitos(P, a.n, ~dup if a.excluir_dups else None)
    # vizinhos = textos ÚNICOS do treino, com a contagem de rótulos que cada um recebeu (cópias entram como contagem)
    grp = tr.groupby("text")["label_id"].apply(list)
    utext = grp.index.tolist()
    ucnt = [{LABELS[c]: l.count(c) for c in range(len(LABELS)) if l.count(c)} for l in grp]
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xu = vec.fit_transform(utext)
    S = (vec.transform(tt.iloc[idx]) @ Xu.T).toarray()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    pist = dict(zip(*[pd.read_csv(a.pistas)[c] for c in ("linha", "pistas")])) if a.pistas else {}
    rows, blocos = [], []
    for j, i in enumerate(idx):
        viz = [(utext[t], ucnt[t], S[j, t]) for t in np.argsort(-S[j])[:a.k]]
        pred = LABELS[int(P[i].argmax())]
        rows.append({"linha": int(i), "margem": round(float(margem[i]), 4), "pred_modelo": pred,
                     **{f"p_{l}": round(float(P[i, c]), 3) for c, l in enumerate(LABELS)}, "texto": tt.iloc[i]})
        b = [f"### CASO {i}\nTEXTO: {tt.iloc[i][:a.max_chars]}\nVIZINHOS DO TREINO (mais parecidos primeiro; [rótulos recebidos x vezes]):"]
        if int(i) in pist: b.insert(1, f"PISTAS LEXICAIS (eli5, só evidência fraca): {pist[int(i)]}")
        b += [f"  - [{', '.join(f'{l}x{n}' for l, n in cnt.items())}] {t[:a.viz_chars]}" for t, cnt, _ in viz]
        blocos.append("\n".join(b))
    pd.DataFrame(rows).to_csv(out / "suspeitos.csv", index=False)
    for s0 in range(0, len(blocos), a.batch):
        (out / f"lote_{s0 // a.batch + 1:02d}.md").write_text("\n\n".join(blocos[s0:s0 + a.batch]), encoding="utf-8")
    print(f"{len(idx)} casos em {out}/suspeitos.csv e {len(range(0, len(blocos), a.batch))} lotes (lote_XX.md)")
    print("Os lotes NÃO trazem a previsão do modelo: só texto e vizinhos, para o LLM julgar sem viés.")


def ler_respostas(f):
    """Lê só as 2 primeiras colunas (linha,rotulo); tolera vírgulas sem aspas no campo 'motivo'."""
    v = {}
    for ln in open(f, encoding="utf-8-sig"):
        p = ln.strip().split(",", 2)
        if len(p) >= 2 and p[0].strip().isdigit() and p[1].strip() in LABELS:
            v[int(p[0])] = LABELS.index(p[1].strip())
    print(f"  {f}: {len(v)} respostas válidas")
    return v


def cmd_apply(a):
    te = pd.read_excel(a.test)
    P = np.load(a.probs)
    idx = pd.read_csv(a.suspeitos)["linha"].astype(int).values
    base = P.argmax(1)
    novo = base.copy()
    votos = []
    for f in a.respostas:
        votos.append(ler_respostas(f))
    trocas = 0
    for i in idx:
        c = Counter(v[int(i)] for v in votos if int(i) in v and v[int(i)] != base[i])
        if c and c.most_common(1)[0][1] >= a.min_votos:
            novo[i] = c.most_common(1)[0][0]
            trocas += 1
    print(f"{len(idx)} suspeitos | {len(a.respostas)} LLMs | {trocas} rótulos trocados (consenso >= {a.min_votos} votos contra o modelo)")
    if a.labels:
        y = pd.read_excel(a.labels)["clarity"].map({l: i for i, l in enumerate(LABELS)}).values
        ant, dep = (base == y), (novo == y)
        print(f"acurácia no teste: antes {ant.mean() * 100:.2f}%  depois {dep.mean() * 100:.2f}%  ({(dep.sum() - ant.sum()):+d} acertos)")
        ch = novo != base
        print(f"entre as {ch.sum()} trocas: {int((~ant & dep)[ch].sum())} errado->certo | {int((ant & ~dep)[ch].sum())} certo->errado | "
              f"{int((~ant & ~dep)[ch].sum())} errado->errado")
        sus = np.zeros(len(y), bool); sus[idx] = True
        print(f"acurácia do modelo nos {len(idx)} suspeitos: {ant[sus].mean() * 100:.1f}%  | com LLMs: {dep[sus].mean() * 100:.1f}%")
        for k, v in enumerate(votos):
            m = [i for i in idx if int(i) in v]
            print(f"  LLM {k + 1}: acerta {np.mean([v[int(i)] == y[i] for i in m]) * 100:.1f}% dos suspeitos que respondeu ({len(m)})")
    te["clarity"] = [LABELS[i] for i in novo]
    te.to_excel(a.out, index=False)
    print("salvo:", a.out)


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("prep")
    q.add_argument("--train", default="train.xlsx"); q.add_argument("--test", required=True); q.add_argument("--probs", required=True)
    q.add_argument("--n", type=int, default=100); q.add_argument("--k", type=int, default=6); q.add_argument("--batch", type=int, default=25)
    q.add_argument("--incluir_dups", dest="excluir_dups", action="store_false", help="inclui cópias exatas do treino nos suspeitos")
    q.add_argument("--max_chars", type=int, default=1500); q.add_argument("--viz_chars", type=int, default=350)
    q.add_argument("--out_dir", default="llm_in"); q.add_argument("--pistas", default=None, help="pistas.csv do eli5_guia.py pistas")
    r = sp.add_parser("apply")
    r.add_argument("--test", required=True); r.add_argument("--probs", required=True); r.add_argument("--suspeitos", required=True, help="suspeitos.csv gerado pelo prep")
    r.add_argument("--respostas", nargs="+", required=True); r.add_argument("--min_votos", type=int, default=2)
    r.add_argument("--labels", default=None); r.add_argument("--out", default="ens_llm.xlsx")
    a = p.parse_args()
    {"prep": cmd_prep, "apply": cmd_apply}[a.cmd](a)


if __name__ == "__main__":
    main()
