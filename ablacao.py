"""
ablacao.py -- ablação geral do ensemble, com QUALQUER conjunto de membros (transformers, HAN, kNN, NB, ...).
Só CPU. Usa as previsões fora da amostra (OOF) de pastas no formato do train_v2.py / members_extra.py /
train_han.py, e a estimativa HONESTA de sempre (pesos ajustados só nos outros folds, por log-loss).

O que imprime:
  1. cada membro sozinho;
  2. ensemble com todos os membros;
  3. LEAVE-ONE-OUT: ensemble sem cada membro (queda = quanto ele contribui), com IC 95% pareado por texto;
  4. ADICIONAR UM: a partir do NÚCLEO (--core), quanto cada candidato acrescenta, com IC 95%;
  5. SELEÇÃO PROGRESSIVA (forward): começa pelo melhor e adiciona o que mais reduz a log-loss aninhada;
  6. concordância par a par e acurácia oráculo;
  7. o comando do ensemble.py para o melhor conjunto.

Cuidado com a leitura: com ~20 mil linhas, diferenças < ~0,3 ponto de acurácia são indistinguíveis de sorte, e
a seleção progressiva escolhe em cima dos mesmos dados (otimista). Prefira conclusões que apareçam também no
leave-one-out e no 'adicionar um', e decida pela log-loss quando a acurácia empatar.

Uso:
    python ablacao.py --data train.xlsx \
        --runs runs_v2/final_large_* runs_v2/final_legal_* runs_v2/final_g_norb_lr3e-5_* runs_v2/HAN_* runs_v2/X_knn_tfidf_* \
        --names BERTimbau LegalNLP NorBERTo HAN kNN-tfidf --add_tfidf --core BERTimbau LegalNLP NorBERTo TF-IDF
"""
import argparse
import glob
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from data_utils import LABELS, load_train
from ensemble import tfidf_member, fit_weights_logloss, fast_acc_ll
from grid3f import paired_bootstrap

COLS = [f"p_{l}" for l in LABELS]


def nested(members, idx, y, fold):
    """Previsões aninhadas do ensemble formado por members[idx]; devolve (P, acc, logloss)."""
    oofs = [members[i] for i in idx]
    P = np.zeros_like(oofs[0])
    for k in np.unique(fold):
        te = fold == k
        w = fit_weights_logloss([O[~te] for O in oofs], y[~te]) if len(oofs) > 1 else np.ones(1)
        P[te] = sum(wi * O[te] for wi, O in zip(w, oofs))
    a, l = fast_acc_ll(P, y)
    return P, a, l


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="train.xlsx")
    p.add_argument("--runs", nargs="+", required=True, help="pastas (aceita glob; vale a mais recente de cada padrão)")
    p.add_argument("--names", nargs="+", required=True, help="um nome por --runs, na mesma ordem")
    p.add_argument("--add_tfidf", action="store_true", help="inclui o TF-IDF+LogReg como membro 'TF-IDF'")
    p.add_argument("--core", nargs="*", default=None, help="nomes do núcleo para o 'adicionar um' (padrão: os 3 primeiros)")
    p.add_argument("--out", default="ablacao.txt")
    a = p.parse_args()
    assert len(a.runs) == len(a.names), "um --names para cada --runs"

    df = load_train(a.data)
    y = df["label_id"].values
    texts = df["text"].values
    names, M, dirs, fold = [], [], {}, None
    for pat, nm in zip(a.runs, a.names):
        found = sorted(glob.glob(pat), key=lambda d: Path(d).stat().st_mtime)
        if not found:
            print(f"[aviso] {pat}: nenhuma pasta encontrada -- {nm} ignorado")
            continue
        d = Path(found[-1])
        o = pd.read_csv(d / "oof_probs.csv")
        P = o[COLS].values.astype(np.float64)
        if len(o) != len(df) or (o["clarity"].values != df["clarity"].values).any():
            print(f"[aviso] {nm} ({d.name}): não bate com {a.data} -- ignorado")
            continue
        if np.isnan(P).any():
            print(f"[aviso] {nm} ({d.name}): OOF incompleto -- ignorado")
            continue
        if fold is None:
            fold = o["fold"].values
        names.append(nm)
        M.append(P / P.sum(1, keepdims=True))
        dirs[nm] = d
    if a.add_tfidf:
        print("[info] calculando o TF-IDF + regressão logística nos mesmos folds...")
        tf, _ = tfidf_member(df, fold, int(fold.max()) + 1, None, "word")
        names.append("TF-IDF")
        M.append(tf.astype(np.float64))
    n = len(M)
    assert n >= 2, "preciso de pelo menos 2 membros"
    core = a.core if a.core else names[:min(3, n)]
    core = [c for c in core if c in names]
    cidx = [names.index(c) for c in core]

    out = []

    def say(s=""):
        print(s, flush=True)
        out.append(s)

    cor = lambda P: P.argmax(1) == y
    say(f"=== ABLAÇÃO | {len(df)} linhas | {n} membros: {', '.join(names)} ===")
    say("Estimativa aninhada: pesos por log-loss ajustados só nos outros folds. Erro-padrão de uma acurácia ~46% "
        "com 20 mil linhas: ~0,35 ponto.\n")

    say("1. CADA MEMBRO SOZINHO")
    for nm, P in zip(names, M):
        acc, ll = fast_acc_ll(P, y)
        say(f"   {nm:16s} acc={acc * 100:6.2f}%  log-loss={ll:.4f}")

    P_all, acc_all, ll_all = nested(M, list(range(n)), y, fold)
    say(f"\n2. ENSEMBLE COM TODOS: acc={acc_all * 100:.2f}%  log-loss={ll_all:.4f}")

    say("\n3. LEAVE-ONE-OUT (ensemble sem o membro; 'perda' > 0 = o membro ajuda)")
    say(f"   {'sem':16s} {'acc':>7s} {'log-loss':>9s}   perda de acurácia vs todos [IC 95%]")
    loo = {}
    for i, nm in enumerate(names):
        idx = [j for j in range(n) if j != i]
        P, ac, ll = nested(M, idx, y, fold)
        loo[nm] = (ac, ll)
        lo, hi = paired_bootstrap(cor(P_all), cor(P), texts)       # todos - sem = quanto o membro agrega
        say(f"   {nm:16s} {ac * 100:6.2f}% {ll:9.4f}   {(acc_all - ac) * 100:+.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}]"
            f"{' *' if lo > 0 else ''}")

    say(f"\n4. ADICIONAR UM AO NÚCLEO ({', '.join(core)})")
    P_core, ac0, ll0 = nested(M, cidx, y, fold)
    say(f"   núcleo sozinho: acc={ac0 * 100:.2f}%  log-loss={ll0:.4f}")
    best_add = []
    for i, nm in enumerate(names):
        if i in cidx:
            continue
        P, ac, ll = nested(M, cidx + [i], y, fold)
        lo, hi = paired_bootstrap(cor(P), cor(P_core), texts)
        say(f"   + {nm:14s} acc={ac * 100:6.2f}%  log-loss={ll:.4f}   {(ac - ac0) * 100:+.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}]"
            f"{' *' if lo > 0 else ''}")
        best_add.append((ll, nm))

    say("\n5. SELEÇÃO PROGRESSIVA (critério: log-loss aninhada; otimista, pois escolhe nos mesmos dados)")
    chosen, rest, path = [], list(range(n)), []
    while rest:
        cands = [(nested(M, chosen + [i], y, fold), i) for i in rest]
        (P, ac, ll), i = min(cands, key=lambda c: c[0][2])
        chosen.append(i)
        rest.remove(i)
        path.append((names[i], ac, ll))
        say(f"   passo {len(chosen)}: + {names[i]:14s} acc={ac * 100:6.2f}%  log-loss={ll:.4f}")
    best_k = int(np.argmin([l for _, _, l in path])) + 1
    best_set = [names[i] for i in chosen[:best_k]]
    say(f"   menor log-loss com {best_k} membro(s): {', '.join(best_set)}")

    say("\n6. CONCORDÂNCIA PAR A PAR DAS PREVISÕES E ORÁCULO")
    pred = [P.argmax(1) for P in M]
    for i, j in itertools.combinations(range(n), 2):
        say(f"   {names[i]:14s} x {names[j]:14s} {(pred[i] == pred[j]).mean() * 100:5.1f}%")
    say(f"   oráculo (pelo menos um acerta): {np.mean(np.any([p_ == y for p_ in pred], axis=0)) * 100:.1f}% | "
        f"todos acertam: {np.mean(np.all([p_ == y for p_ in pred], axis=0)) * 100:.1f}%")

    runs_cmd = [str(dirs[nm]) for nm in best_set if nm in dirs]
    flags = " --add_tfidf" if "TF-IDF" in best_set else ""
    say("\n7. COMANDO PARA O MELHOR CONJUNTO (confira com a leitura dos itens 3 e 4, não só com o 5):")
    say(f"   python ensemble.py --data {a.data} --test test.xlsx{flags} --runs {' '.join(runs_cmd)} --out ens_ablacao.xlsx")
    Path(a.out).write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
