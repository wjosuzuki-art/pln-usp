# EP1 – Classificação da clareza de respostas do e-SIC

Exercício prático 1 de ACH2118 – Introdução ao Processamento de Língua Natural (EACH-USP).
Tarefa: classificar cada resposta do e-SIC em **c1**, **c234** ou **c5** (nota de clareza dada pelo cidadão, com as notas 2, 3 e 4 agrupadas).

## Resultado

Modelo final: **ensemble de quatro classificadores**, combinados por média ponderada das probabilidades.

| Componente | Representação | Acurácia (5 folds) |
|---|---|---|
| BERTimbau-large | ajuste fino (fine-tuning) | 45,54% |
| LegalNLP-BERT | ajuste fino | 45,58% |
| NorBERTo-large | ajuste fino | 45,85% |
| TF-IDF + regressão logística (baseline) | palavras, 1–2 gramas | 45,18% |
| **Ensemble** (pesos 0,36 / 0,19 / 0,11 / 0,34) | média ponderada | **47,34%** |

Validação cruzada de 5 folds agrupada por texto, com os pesos de cada fold ajustados só nos outros quatro folds.
O ensemble supera o baseline nos 5 folds (+1,67 a +2,66 pontos percentuais; média +2,16).
É uma estimativa de validação: os rótulos do conjunto de teste não são conhecidos. Detalhes e tabelas no relatório.

## Estrutura do repositório

| Arquivo | Para que serve | No modelo final |
|---|---|---|
| `data_utils.py` | carga e limpeza dos dados, divisões agrupadas por texto | sim |
| `train_v2.py` | treino dos transformers (loop próprio, 5 folds, alvos suaves para duplicatas, previsões fora da amostra e de teste) | sim |
| `ensemble.py` | combina os modelos, ajusta os pesos, estimativa honesta aninhada, gera a planilha de teste | sim |
| `predict_v2.py` | prevê o teste a partir dos modelos salvos (se o treino foi feito sem `--test`) | opcional |
| `grid_v2.py`, `grid_novos.py` | triagem de hiperparâmetros (1 fold por configuração) | seleção |
| `analise_ensemble.py` | tabelas do relatório (por fold, ablação, concordância, pesos) e `fold_assignments.csv` | análise |
| `por_classe.py`, `marginal_fold0.py`, `compare_oof.py` | análise por classe, valor marginal de cada modelo, comparação de duas rodadas | análise |
| `train_transformer.py`, `train_setfit.py`, `predict_test.py` | primeira versão do pipeline e experimentos iniciais | não |
| `tapt.py`, `extract_v1_oof.py`, `fs_extratrees.py` | pré-treino no domínio, extração de previsões antigas, seleção de atributos (testes descartados ou não usados) | não |
| `fold_assignments.csv` | fold de cada linha do treino (`linha_planilha` = linha no Excel), para reproduzir a validação exatamente | sim |
| `requirements.txt`, `requirements-lock.txt` | dependências (as principais fixadas / ambiente completo) | – |
| `resultados/` | placares das triagens, `results_log.csv`, saída do ensemble e das análises | – |

## Ambiente

Testado em: Linux, NVIDIA RTX 3060 (12 GB), Python 3.12.3, PyTorch 2.14.0 (CUDA 12.6), transformers 5.17.0, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.3, scipy 1.18.1, openpyxl 3.1.5.
Os transformers precisam de GPU (os tempos abaixo são da RTX 3060); o ensemble e as análises rodam em CPU.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Para reproduzir o ambiente completo, use `requirements-lock.txt` (gerado com `pip freeze`).

## Dados

Os dados da disciplina **não** estão neste repositório. Copie `train.xlsx` e `test.xlsx` para a pasta raiz.
Colunas esperadas: `resp_text` e `clarity` (no teste, `clarity` vazia).
Uma linha do treino (a linha 5.028 da planilha, cujo texto é só o número `23500000000000000`) é descartada na carga: o treino usa 20.091 das 20.092 linhas.

## Reprodução passo a passo

**1. Treinar os três transformers** (5 folds cada; gera as previsões fora da amostra e as do teste).
Tempos na RTX 3060: ~179 min, ~56 min e ~218 min.

```bash
python train_v2.py --data train.xlsx --test test.xlsx --kfold 5 --tag final_large \
  --model neuralmind/bert-large-portuguese-cased --lr 2e-5 --epochs 3 --batch_size 8 --grad_accum 2 --max_len 512

python train_v2.py --data train.xlsx --test test.xlsx --kfold 5 --tag final_legal \
  --model felipemaiapolo/legalnlp-bert --lr 3e-5 --epochs 3 --batch_size 16 --grad_accum 1 --max_len 512

python train_v2.py --data train.xlsx --test test.xlsx --kfold 5 --tag final_norb \
  --model Itau-Unibanco/NorBERTo-large --lr 3e-5 --epochs 3 --batch_size 8 --grad_accum 2 --max_len 512
```

Os demais parâmetros usam os padrões de `train_v2.py` (AdamW, warmup de 10%, weight decay 0,01, decaimento por camada 0,9, label smoothing 0,05, peso √n para duplicatas, bf16, semente 42) e ficam registrados em `config.json` dentro de cada pasta de resultados (`runs_v2/<tag>_<timestamp>/`).

**2. Combinar os modelos e gerar a planilha de teste rotulada:**

```bash
python ensemble.py --data train.xlsx --test test.xlsx --add_tfidf --full_mode mix --runs runs_v2/final_large_* runs_v2/final_legal_* runs_v2/final_norb_* --out submission_final.xlsx
```

**2b. Treina e combina modelos com dados de toda a planilha train.xlsx:**

```bash
python train_v2.py --data train.xlsx --test test.xlsx --full --full_into runs_v2/final_large_* --tag full_large --model neuralmind/bert-large-portuguese-cased --lr 2e-5 --epochs 3 --batch_size 8 --grad_accum 2 --max_len 512
python train_v2.py --data train.xlsx --test test.xlsx --full --full_into runs_v2/final_legal_* --tag full_legal --model felipemaiapolo/legalnlp-bert --lr 3e-5 --epochs 3 --batch_size 16 --grad_accum 1 --max_len 512
python train_v2.py --data train.xlsx --test test.xlsx --full --full_into runs_v2/final_norb_* --tag full_norb --model Itau-Unibanco/NorBERTo-large --lr 3e-5 --epochs 3 --batch_size 8 --grad_accum 2 --max_len 512
python ensemble.py --data train.xlsx --test test.xlsx --add_tfidf --full_mode mix --runs runs_v2/final_large_* runs_v2/final_legal_* runs_v2/final_norb_* --out submission_final.xlsx
```

Imprime a comparação dos métodos de pesos, os pesos escolhidos e a **estimativa honesta** (47,34%). Gera `submission_final.xlsx` (a entrega) e `submission_final_dupblend.xlsx` (variante descartada, não validada).

**3. Tabelas de análise do relatório** (por fold, ablação, concordância entre modelos, pesos por fold):

```bash
python analise_ensemble.py --data train.xlsx --test test.xlsx \
  --runs runs_v2/final_large_* runs_v2/final_legal_* runs_v2/final_norb_* \
  --names BERTimbau-large LegalNLP-BERT NorBERTo-large
```

### Triagem de hiperparâmetros

```bash
python grid_v2.py --data train.xlsx        # BERTimbau-large/base, LegalNLP-BERT (1 fold por configuração)
python grid_novos.py --data train.xlsx     # NorBERTo-large e Albertina-900M com LoRA
```

Os placares estão em `resultados/`. Como cada configuração foi avaliada em um fold só (erro-padrão de cerca de 0,8 ponto), pequenas diferenças entre elas não são conclusivas.

```bash

```

Não há validação independente para esses modelos; os 47,34% referem-se aos modelos dos folds.

## Arquivos gerados em cada rodada (`runs_v2/<tag>_<timestamp>/`)

`config.json` (parâmetros), `summary.json` (métricas por fold e totais), `oof_probs.csv` (probabilidades fora da amostra, usadas pelo ensemble), `test_probs_mean.npy` (média dos 5 modelos no teste), `test_probs_fold*.npy`, `model_fold*/` (com `--save_models`).

## Problemas conhecidos

- **PyTorch com CUDA não instala no Python 3.14** (no Windows): use Python 3.12.
- **`Driver/library version mismatch` no `nvidia-smi`:** costuma resolver reiniciando a máquina.
- **`CUDA out of memory`:** reduza `--batch_size` e aumente `--grad_accum` na mesma proporção (ex.: `--batch_size 4 --grad_accum 4`).
- **Pesos `.bin` bloqueados pelo transformers:** exigem torch ≥ 2.6; o `train_v2.py` tenta `safetensors` primeiro.
- **A divisão do `GroupKFold` pode variar entre versões do scikit-learn:** `fold_assignments.csv` fixa a divisão usada.

## Limitações

A triagem foi feita em um único fold; o efeito isolado do agrupamento de duplicatas não foi medido por ablação controlada; os testes de pré-processamento usaram TF-IDF; e textos praticamente idênticos têm o mesmo rótulo em apenas 53,7% dos casos (ruído de anotação), o que limita a acurácia de qualquer modelo. Ver o relatório.
