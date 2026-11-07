# EP2 – Clareza de respostas do e-SIC (ACH2118 – Introdução ao PLN)

Classificação da clareza de respostas a pedidos de acesso à informação em {c1, c234, c5}.
Integrantes: Caique Alves de Souza, Gabriela Alcaide, Gustavo Pompermayer, William Suzuki.

## Modelo final
Ensemble de 9 classificadores (média ponderada das probabilidades, pesos por log-loss): BERTimbau-large, LegalNLP-BERT,
NorBERTo-large (fine-tuning, 5 folds), TF-IDF + regressão logística, kNN-TF-IDF, Naive Bayes, HGB sobre estatísticas do texto,
kNN sobre embeddings e5 e HAN. Validação (OOF, 5 folds): 47,43%. Teste 1: 51,22% (EP1: 49,89%; McNemar p ≈ 0,10, não significativo).

## Arquivos principais
- `train_v2.py`, `predict_v2.py`: fine-tuning dos transformers e previsão (média dos modelos de fold).
- `grid3f.py`, `grid_ens.py`, `triage_ptt5.py`: grids de hiperparâmetros e triagem do PTT5.
- `members_extra.py`, `train_han.py`: membros adicionais (kNN, NB, HGB, embeddings, HAN).
- `ablacao.py`: ablação do ensemble. `ensemble.py`: ensemble final. `run_ep2_9.sh`: pipeline do ensemble de 9 membros.
- `score_test1.py`, `mcnemar_ep.py`: avaliação no teste 1 e teste de McNemar contra o EP1.
- `llm_consenso.py`, `eli5_guia.py`, `prior_shift.py`: experimentos de correção pós-modelo (resultados negativos) e guia eli5.

## Como reproduzir
1. `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` (GPU com ~12 GB).
2. Colocar `train.xlsx`, `test.xlsx` (teste 1), `test2.xlsx` na pasta (os dados do curso não estão neste repositório).
3. Treinar os transformers em 5 folds: `python train_v2.py --kfold 5 --save_models --tag final_...` (receitas no relatório, seção 5).
4. Gerar as previsões de teste com `predict_v2.py` e rodar `bash run_ep2_9.sh`.
5. Avaliar: `python score_test1.py ...` e `python mcnemar_ep.py ...`.
