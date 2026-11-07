#!/bin/bash
cd /projetos/texto && source .venv/bin/activate
R1=runs_v2/final_large_1790601626; R2=runs_v2/final_legal_1790612371; R3=runs_v2/final_g_norb_lr3e-5_1790685940
L=test1-rotulado-para-teste-estatistico.xlsx
echo "=== [0] conferindo ==="
md5sum -c --quiet <<'SUM' || { echo "ARQUIVO DIFERENTE -- copie de novo"; exit 1; }
c9923155a86c244d4dd7b048b7c387d2  ensemble.py
de944cc2c69b9a93d11037d16f5850cd  llm_consenso.py
c2b03ed310ff7b8d2cdce3de777c4456  mcnemar_ep.py
SUM
for f in $L test.xlsx test2.xlsx train.xlsx; do [ -f "$f" ] || { echo "FALTA $f nesta pasta"; exit 1; }; done
echo "=== [1] guarda/restaura as previsões do test1 ==="
for d in $R1 $R2 $R3; do [ -f $d/test_probs_mean_test1.npy ] || cp $d/test_probs_mean.npy $d/test_probs_mean_test1.npy; cp $d/test_probs_mean_test1.npy $d/test_probs_mean.npy; done
echo "=== [2] ensemble de 4 membros no test1 ==="
python ensemble.py --data train.xlsx --test test.xlsx --add_tfidf --runs $R1 $R2 $R3 --out ens_core_t1.xlsx || exit 1
echo "=== [3] acurácia no test1 (só modelos treinados SEM o test1) ==="
python score_test1.py --labels $L --runs runs_v2/final_* runs_v2/X_* runs_v2/HAN_* --subs "ens_core_t1*.xlsx" "ens_ablacao*.xlsx" | tee score_test1.txt
echo "=== [4] previsões do test2 (3 famílias) e ensemble ==="
for d in $R1 $R2 $R3; do python predict_v2.py --run $d --data train.xlsx --test test2.xlsx || exit 1; cp $d/test_probs_mean.npy $d/test_probs_mean_test2.npy; done
python ensemble.py --data train.xlsx --test test2.xlsx --add_tfidf --runs $R1 $R2 $R3 --out entrega_test2.xlsx || exit 1
for d in $R1 $R2 $R3; do cp $d/test_probs_mean_test1.npy $d/test_probs_mean.npy; done
echo "=== [5] confere o formato da entrega ==="
python - <<'PY'
import pandas as pd
t=pd.read_excel("test2.xlsx"); 
for f in ("entrega_test2.xlsx","entrega_test2_dupblend.xlsx"):
    e=pd.read_excel(f)
    print(f, "linhas", len(t), len(e), "| mesmas colunas:", list(t.columns)==list(e.columns), "| textos na mesma ordem:", (t["resp_text"].astype(str).values==e["resp_text"].astype(str).values).all(),
          "| vazios:", int(e["clarity"].isna().sum()), "|", e["clarity"].value_counts(normalize=True).round(3).to_dict())
PY
echo "=== FIM ==="
