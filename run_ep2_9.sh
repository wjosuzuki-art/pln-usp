#!/bin/bash
# Ensemble de 9 membros (EP2) para test1 + test2 de uma vez, em um arquivo combinado (900 + 897 linhas).
cd /projetos/texto && source .venv/bin/activate
F='runs_v2/final_large_*'
R1=runs_v2/final_large_1790601626; R2=runs_v2/final_legal_1790612371; R3=runs_v2/final_g_norb_lr3e-5_1790685940
for d in $R1 $R2 $R3; do for t in test1 test2; do [ -f $d/test_probs_mean_$t.npy ] || { echo "FALTA $d/test_probs_mean_$t.npy -- rode antes o run_ep2.sh até o fim"; exit 1; }; done; done

echo "=== [1] arquivo combinado test12.xlsx e previsões dos transformers concatenadas ==="
python - <<'PY' || exit 1
import numpy as np, pandas as pd
a, b = pd.read_excel("test.xlsx"), pd.read_excel("test2.xlsx")
for d in (a, b):
    if "clarity" not in d: d["clarity"] = np.nan
pd.concat([a[["resp_text", "clarity"]], b[["resp_text", "clarity"]]], ignore_index=True).to_excel("test12.xlsx", index=False)
print("test12.xlsx:", len(a), "+", len(b), "linhas")
for d in ("runs_v2/final_large_1790601626", "runs_v2/final_legal_1790612371", "runs_v2/final_g_norb_lr3e-5_1790685940"):
    p1, p2 = np.load(f"{d}/test_probs_mean_test1.npy"), np.load(f"{d}/test_probs_mean_test2.npy")
    assert len(p1) == len(a) and len(p2) == len(b), (d, p1.shape, p2.shape)
    np.save(f"{d}/test_probs_mean.npy", np.concatenate([p1, p2]))
PY

echo "=== [2] membros extras e HAN com test12 (usam os mesmos folds do final_large) ==="
python members_extra.py --data train.xlsx --test test12.xlsx --folds_from "$F" --members knn_grid nb_tfidf hgb_stats knn_emb || exit 1
python train_han.py --data train.xlsx --test test12.xlsx --folds_from "$F" --tag HANt12 || exit 1
KG=$(ls -dt runs_v2/X_knn_grid_*|head -1); NB=$(ls -dt runs_v2/X_nb_tfidf_*|head -1); HG=$(ls -dt runs_v2/X_hgb_stats_*|head -1); KE=$(ls -dt runs_v2/X_knn_emb_*|head -1); HN=$(ls -dt runs_v2/HANt12_*|head -1)
echo "usando: $KG $NB $HG $KE $HN"

echo "=== [3] ensemble de 9 membros em test12 ==="
python ensemble.py --data train.xlsx --test test12.xlsx --add_tfidf --runs $R1 $R2 $R3 $KG $NB $HG $KE $HN --out ens9_t12.xlsx || exit 1

echo "=== [4] separa test1 (900) e test2 (897) ==="
python - <<'PY' || exit 1
import pandas as pd
t1, t2 = pd.read_excel("test.xlsx"), pd.read_excel("test2.xlsx")
for suf in ("", "_dupblend"):
    e = pd.read_excel(f"ens9_t12{suf}.xlsx")
    assert len(e) == len(t1) + len(t2)
    a, b = e.iloc[:len(t1)].reset_index(drop=True), e.iloc[len(t1):].reset_index(drop=True)
    assert (a["resp_text"].astype(str).values == t1["resp_text"].astype(str).values).all()
    assert (b["resp_text"].astype(str).values == t2["resp_text"].astype(str).values).all()
    a.to_excel(f"ens9_t1{suf}.xlsx", index=False); b.to_excel(f"entrega9_test2{suf}.xlsx", index=False)
    print(f"ens9_t1{suf}.xlsx ({len(a)}) | entrega9_test2{suf}.xlsx ({len(b)}) | clarity vazios:", int(b['clarity'].isna().sum()), "| colunas:", list(b.columns) == list(t2.columns))
PY
echo "=== [5] acurácia no test1 ==="
python score_test1.py --labels test1-rotulado-para-teste-estatistico.xlsx --runs $R1 --subs "ens9_t1*.xlsx" "ens_core_t1*.xlsx" | grep -E "xlsx|test1:"
echo "=== FIM ==="
