"""
predict_v2.py -- gera a planilha de entrega a partir de uma rodada do
train_v2.py que foi executada com --save_models. Não treina nada: só
carrega os modelos salvos de cada fold, prevê o teste, tira a média
entre os folds e escreve os arquivos de submissão.

Uso (quando o test.xlsx for liberado):
    python predict_v2.py --run runs_v2/bertlarge_1790000000 --data train.xlsx --test test.xlsx

Saída, dentro da própria pasta da rodada:
    submission_model.xlsx     -- só o modelo
    submission_dupblend.xlsx  -- modelo + rótulos de textos idênticos do treino
    test_probs_mean.npy       -- probabilidades (usadas pelo ensemble.py)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_utils import load_train, clean_text
from train_v2 import (encode_head_tail, predict_probs, setup_precision,
                      dup_blend, write_submission)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="Pasta da rodada do train_v2.py (com model_fold*/)")
    p.add_argument("--data", default="train.xlsx", help="Usado só para o resgate de duplicatas")
    p.add_argument("--test", required=True)
    p.add_argument("--eval_batch_size", type=int, default=32)
    args = p.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    run = Path(args.run)
    cfg = json.load(open(run / "config.json"))
    fold_dirs = sorted(run.glob("model_fold*"))
    assert fold_dirs, (f"Nenhum model_fold* em {run}. A rodada foi feita com --save_models?")
    print(f"[info] {len(fold_dirs)} modelos encontrados: {[d.name for d in fold_dirs]}")

    tdf = pd.read_excel(args.test)
    assert "resp_text" in tdf.columns, f"test sem coluna resp_text: {tdf.columns.tolist()}"
    test_texts = tdf["resp_text"].apply(clean_text).tolist()

    device, amp_dtype = setup_precision(torch)
    tokenizer = AutoTokenizer.from_pretrained(fold_dirs[0])
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    seqs = encode_head_tail(tokenizer, test_texts, cfg["max_len"], cfg["head_tokens"])

    all_probs = []
    for d in fold_dirs:
        model = AutoModelForSequenceClassification.from_pretrained(d).to(device)
        probs = predict_probs(torch, model, seqs, pad_id, args.eval_batch_size, device, amp_dtype)
        np.save(run / f"test_probs_{d.name}.npy", probs)
        all_probs.append(probs)
        print(f"[info] {d.name}: previsão feita")
        del model
        torch.cuda.empty_cache()

    mean_probs = np.mean(all_probs, axis=0)
    np.save(run / "test_probs_mean.npy", mean_probs)
    print("submission_model.xlsx:", write_submission(args.test, mean_probs, run / "submission_model.xlsx"))

    train_df = load_train(args.data)
    blended, hits = dup_blend(test_texts, train_df, mean_probs, cfg.get("dup_blend_alpha", 3.0))
    print(f"submission_dupblend.xlsx ({hits} textos idênticos a algum do treino):",
          write_submission(args.test, blended, run / "submission_dupblend.xlsx"))


if __name__ == "__main__":
    main()