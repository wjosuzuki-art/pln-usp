#!/bin/bash
cd /projetos/texto && source .venv/bin/activate
F='runs_v2/final_large_*'
pronto() { ls runs_v2/$1_*/summary.json >/dev/null 2>&1; }   # já existe run concluído?

echo "=== [0] conferindo arquivos ==="
md5sum -c --quiet <<'SUM' || { echo "ARQUIVO DIFERENTE/TRUNCADO -- copie de novo"; exit 1; }
d9cae192e64c1b4407db401fb954ecd5  train_han.py
df27b30a48fbc9d9a6bb84fc54a4b329  members_extra.py
e29300a3fd711c60a04c7c91b71c6d3e  ablacao.py
SUM
python -m py_compile train_han.py members_extra.py ablacao.py || exit 1
pip install -q sentence-transformers

echo "=== [1] membros extras (pula os que já terminaram) ==="
for m in knn_tfidf knn_grid nb_tfidf hgb_stats knn_emb lr_emb; do
  if pronto X_$m; then echo "[pulando] $m já existe"; continue; fi
  python members_extra.py --data train.xlsx --test test.xlsx --folds_from "$F" --members $m \
    || echo "!!! $m falhou -- sigo sem ele"
done

echo "=== [2] HAN (debug, depois 5 folds; pula se já terminou) ==="
if pronto HAN; then echo "[pulando] HAN já existe"; else
  python train_han.py --data train.xlsx --test test.xlsx --debug \
   && python train_han.py --data train.xlsx --test test.xlsx --folds_from "$F" --tag HAN \
   || echo "!!! HAN falhou -- sigo sem ele"
fi

echo "=== [3] ablacao ==="
python ablacao.py --data train.xlsx \
  --runs "runs_v2/final_large_*" "runs_v2/final_legal_*" "runs_v2/final_g_norb_lr3e-5_*" "runs_v2/HAN_*" \
         "runs_v2/X_knn_tfidf_*" "runs_v2/X_knn_grid_*" "runs_v2/X_nb_tfidf_*" "runs_v2/X_hgb_stats_*" "runs_v2/X_knn_emb_*" "runs_v2/X_lr_emb_*" \
  --names BERTimbau LegalNLP NorBERTo HAN kNN-tfidf kNN-grid NB-tfidf HGB-stats kNN-emb LR-emb \
  --add_tfidf --core BERTimbau LegalNLP NorBERTo TF-IDF --out ablacao2.txt || exit 1

echo "=== [4] ensemble com o conjunto sugerido + nota no test1 ==="
CMD=$(grep -m1 "python ensemble.py" ablacao2.txt | sed 's/^ *//')
echo "$CMD"; eval "$CMD" && python score_test1.py --labels test1-rotulado-para-teste-estatistico.xlsx --subs "ens_ablacao.xlsx"
echo "=== FIM ==="
