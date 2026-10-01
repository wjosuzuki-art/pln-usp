"""
train_setfit.py
Alternativa rápida ao fine-tuning completo: SetFit faz fine-tuning
contrastivo de um sentence-transformer pequeno + uma cabeça de
classificação leve por cima. É bem mais barato que treinar um BERT
inteiro (poucos minutos até em CPU para textos como os do e-SIC),
então serve tanto como (a) comparação honesta no relatório -- "testamos
dois métodos modernos, não só um" -- quanto como (b) plano B se o
BERTimbau completo estiver demorando demais na GTX 1060.

Uso:
    python train_setfit.py --data train_full.xlsx --mode split
    python train_setfit.py --data train_full.xlsx --mode kfold --kfold 3

Requer: pip install setfit sentence-transformers
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from data_utils import LABELS, load_train, report_duplicate_leakage_risk, group_train_val_split, group_kfold_splits

RESULTS_LOG = "results_log.csv"  # mesmo arquivo do train_transformer.py -- assim os 3
                                   # métodos (nominal, ordinal, SetFit) ficam lado a lado
                                   # numa única tabela pra comparar no relatório.


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="train_full.xlsx")
    p.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    p.add_argument("--mode", choices=["split", "kfold"], default="split")
    p.add_argument("--kfold", type=int, default=3)
    p.add_argument("--val_size", type=float, default=0.15)
    p.add_argument("--num_pairs", type=int, default=20,
                    help="Pares contrastivos gerados por exemplo de treino (parâmetro central do SetFit).")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def run_one_split(train_df, val_df, args):
    from setfit import SetFitModel, Trainer, TrainingArguments
    from datasets import Dataset

    train_ds = Dataset.from_pandas(train_df[["text", "clarity"]].rename(columns={"clarity": "label"}))
    val_ds = Dataset.from_pandas(val_df[["text", "clarity"]].rename(columns={"clarity": "label"}))

    model = SetFitModel.from_pretrained(args.model)

    targs = TrainingArguments(
        num_epochs=args.epochs,
        num_iterations=args.num_pairs,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        column_mapping={"text": "text", "label": "label"},
    )
    trainer.train()

    preds = model.predict(val_df["text"].tolist())
    preds = np.asarray(preds)
    labels = val_df["clarity"].values

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    print(classification_report(labels, preds, target_names=LABELS, digits=3))
    print("Matriz de confusão:\n", LABELS)
    print(confusion_matrix(labels, preds, labels=LABELS))
    return acc, f1


def log_result(run_name, args, acc, f1):
    # mesmas colunas do train_transformer.py -- campos que não se aplicam ao
    # SetFit (lr/batch_size/grad_accum/ordinal_loss) ficam em branco, e o nome
    # do método vai em "model" pra identificar a linha na tabela comparativa.
    header = ["run_name", "model", "mode", "lr", "epochs", "batch_size", "grad_accum",
              "ordinal_loss", "accuracy", "macro_f1"]
    row = [run_name, f"SetFit:{args.model}", args.mode, "", args.epochs, "", "",
           "", acc, f1]
    write_header = not Path(RESULTS_LOG).exists()
    with open(RESULTS_LOG, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)


def main():
    args = parse_args()
    df = load_train(args.data)
    report_duplicate_leakage_risk(df)

    if args.debug:
        df = df.sample(min(300, len(df)), random_state=args.seed).reset_index(drop=True)
        print(f"[debug] usando apenas {len(df)} linhas")

    t0 = time.time()

    if args.mode == "split":
        train_df, val_df = group_train_val_split(df, val_size=args.val_size, seed=args.seed)
        acc, f1 = run_one_split(train_df, val_df, args)
        print(f"\nSetFit (split único): acc={acc:.4f}  macro-F1={f1:.4f}")
        log_result(f"setfit_split_{int(t0)}", args, acc, f1)
    else:
        accs, f1s = [], []
        for i, (train_df, val_df) in enumerate(group_kfold_splits(df, n_splits=args.kfold, seed=args.seed)):
            print(f"\n===== FOLD {i+1}/{args.kfold} =====")
            acc, f1 = run_one_split(train_df, val_df, args)
            accs.append(acc)
            f1s.append(f1)
            log_result(f"setfit_kfold{i}_{int(t0)}", args, acc, f1)
        print(f"\nMédia {args.kfold}-fold: acc={np.mean(accs):.4f} (+-{np.std(accs):.4f})  "
              f"macro-F1={np.mean(f1s):.4f} (+-{np.std(f1s):.4f})")

    print(f"[info] tempo total: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()