"""
train_transformer.py
Fine-tuning de um transformer em português para a classificação
{c1, c234, c5} de clareza de respostas e-SIC (EP1 - ACH2118).

Pensado para RODAR NA SUA MÁQUINA (GTX 1060 / Ryzen 5900X) -- este
sandbox não tem acesso ao Hugging Face Hub para baixar os pesos.

Uso típico (busca rápida de hiperparâmetros, split único):
    python train_transformer.py --data train.xlsx --mode split \
        --model neuralmind/bert-base-portuguese-cased \
        --lr 2e-5 --epochs 3

Uso para o número final do relatório (k-fold agrupado, mais lento):
    python train_transformer.py --data train.xlsx --mode kfold --kfold 3 \
        --model neuralmind/bert-base-portuguese-cased \
        --lr 2e-5 --epochs 3

(batch_size, grad_accum e gradient checkpointing já vêm com defaults
pensados para uma GTX 1060 de 3GB -- só mexa neles se dermos OOM ou se
sobrar VRAM, ver dicas abaixo.)

Dica de hardware (GTX 1060 **3GB** -- confirmado via nvidia-smi, com ~1GB
já ocupado pelo Windows, sobrando ~2GB livres):
  - batch_size=2 + grad_accum=16 (batch efetivo 32) + gradient checkpointing
    ligado por padrão são os valores default agora, pensados pra caber
    nesse espaço apertado. Gradient checkpointing troca ~20-30% de tempo
    de treino por bem menos uso de memória (recalcula ativações no
    backward em vez de guardá-las). Se mesmo assim der CUDA OutOfMemory,
    tente --max_len 128 (corta mais o texto) ou um modelo menor (ver abaixo).
  - Se, depois de rodar, você notar que sobra VRAM (ex.: nvidia-smi mostrando
    bem menos que os 3GB em uso durante o treino), pode tentar
    --batch_size 4 --grad_accum 8 e/ou --no_grad_checkpointing para
    acelerar um pouco.
  - fp16 é detectado e LIGADO SÓ em GPUs com Tensor Cores (compute
    capability >= 7, Volta/Turing/Ampere+). Numa GTX 1060 (Pascal de
    consumidor) ele fica desligado automaticamente, porque lá o cálculo
    em fp16 roda a 1/64 da velocidade do fp32 -- ligaria só pra deixar
    mais lento. Nada a configurar aqui, é automático.
  - Para trocar por um modelo mais leve/rápido (recomendado se OOM
    persistir mesmo com os ajustes acima), use --model com outro nome
    do Hugging Face Hub, ex.: --model adalbertojunior/distilbert-portuguese-cased
    (~40% menor que o BERTimbau-base).
  - Use --debug para rodar em ~300 linhas só para validar que o pipeline
    não quebra antes de comprometer a noite inteira de treino.
"""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from data_utils import (
    LABELS, LABEL2ID, ID2LABEL,
    load_train, report_duplicate_leakage_risk,
    group_train_val_split, group_kfold_splits, duplicate_noise_ceiling,
)

RESULTS_LOG = "results_log.csv"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="train_full.xlsx")
    p.add_argument("--model", default="neuralmind/bert-base-portuguese-cased")
    p.add_argument("--mode", choices=["split", "kfold"], default="split")
    p.add_argument("--kfold", type=int, default=3)
    p.add_argument("--val_size", type=float, default=0.15)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--no_grad_checkpointing", action="store_true",
                    help="Desliga o gradient checkpointing. Ele economiza bastante VRAM "
                         "recalculando ativações no backward em vez de guardá-las, ao custo "
                         "de ~20-30%% mais tempo de treino. Fica ligado por padrão porque sua "
                         "GTX 1060 é a variante de 3GB -- só desligue se confirmar que sobra "
                         "memória de sobra sem ele.")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ordinal_loss", action="store_true",
                    help="Usa Ordinal Log-Loss em vez de cross-entropy padrão "
                         "(trata c1<c234<c5 como uma ordem). Experimental -- "
                         "compare com a versão nominal antes de decidir qual reportar.")
    p.add_argument("--debug", action="store_true", help="Roda com uma amostra pequena p/ testar o pipeline.")
    p.add_argument("--out_dir", default="runs")
    p.add_argument("--tag", default="", help="Rótulo curto pra identificar essa rodada no results_log.csv "
                                               "(ex.: 'lr2e-5_ep4'). Usado pelo run_overnight.py para "
                                               "saber qual linha do CSV pertence a qual configuração.")
    return p.parse_args()


def build_trainer(model, tokenizer, train_ds, val_ds, args, run_name):
    import torch
    from transformers import TrainingArguments, Trainer

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro"),
        }

    # fp16 (mixed precision) só compensa em GPUs com Tensor Cores (Volta/Turing/
    # Ampere em diante, compute capability >= 7.0). Em Pascal "de consumidor"
    # (GTX 10xx, incluindo a 1060 -- diferente da P100, que é Pascal de
    # datacenter com fp16 full-rate), o cálculo em fp16 roda a 1/64 da
    # velocidade do fp32 por falta de Tensor Cores, então ligar isso pode
    # deixar o treino MAIS LENTO em vez de mais rápido. Detectamos e desligamos
    # automaticamente nesse caso.
    use_fp16 = False
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        use_fp16 = major >= 7
        if not use_fp16:
            print(f"[info] GPU com compute capability {major}.x (Pascal ou anterior) "
                  f"-- fp16 desligado automaticamente, vai rodar em fp32.")

    targs = TrainingArguments(
        output_dir=os.path.join(args.out_dir, run_name),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(4, args.batch_size * 2),
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=not args.no_grad_checkpointing,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        fp16=use_fp16,
        logging_steps=50,
        report_to=[],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    return trainer


def make_dataset(df, tokenizer, max_len):
    from datasets import Dataset

    ds = Dataset.from_pandas(df[["text", "label_id"]].rename(columns={"label_id": "labels"}))

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len, padding="max_length")

    ds = ds.map(tok, batched=True, remove_columns=["text"])
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds


def maybe_wrap_ordinal(model, use_ordinal: bool):
    """Troca a loss padrão (cross-entropy) por Ordinal Log-Loss quando
    --ordinal_loss é passado. Mantém a mesma cabeça de 3 saídas -- só muda
    como o erro é penalizado (confundir c1<->c5 pesa mais que c1<->c234)."""
    if not use_ordinal:
        return model
    import torch
    import torch.nn as nn

    n_classes = model.config.num_labels
    # matriz de distância ordinal |i-j| entre classes, usada para ponderar a OLL
    dist = torch.tensor([[abs(i - j) for j in range(n_classes)] for i in range(n_classes)],
                         dtype=torch.float)

    orig_forward = model.forward

    def forward_with_oll(input_ids=None, attention_mask=None, labels=None, **kwargs):
        out = orig_forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        if labels is not None:
            logp = torch.log_softmax(out.logits, dim=-1)
            w = dist.to(out.logits.device)[labels]  # (batch, n_classes) pesos por exemplo
            loss = -(w * logp).sum(dim=-1).mean()
            out.loss = loss
        return out

    model.forward = forward_with_oll
    return model


def run_one_split(train_df, val_df, args, run_name):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(LABELS), use_safetensors=True
    )
    model = maybe_wrap_ordinal(model, args.ordinal_loss)

    train_ds = make_dataset(train_df, tokenizer, args.max_len)
    val_ds = make_dataset(val_df, tokenizer, args.max_len)

    trainer = build_trainer(model, tokenizer, train_ds, val_ds, args, run_name)
    trainer.train()
    metrics = trainer.evaluate()

    preds = np.argmax(trainer.predict(val_ds).predictions, axis=-1)
    labels = val_df["label_id"].values
    report = classification_report(labels, preds, target_names=LABELS, digits=3)
    cm = confusion_matrix(labels, preds)

    print(report)
    print("Matriz de confusão (linhas=real, colunas=previsto):\n", LABELS)
    print(cm)

    return metrics, report, cm


def log_result(run_name, args, metrics):
    header = ["run_name", "model", "mode", "lr", "epochs", "batch_size", "grad_accum",
              "ordinal_loss", "accuracy", "macro_f1"]
    row = [run_name, args.model, args.mode, args.lr, args.epochs, args.batch_size,
           args.grad_accum, args.ordinal_loss,
           metrics.get("eval_accuracy"), metrics.get("eval_macro_f1")]
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
    print(f"[info] teto otimista por ruído de rótulo neste corpus: {duplicate_noise_ceiling(df):.3f}")

    if args.debug:
        df = df.sample(min(300, len(df)), random_state=args.seed).reset_index(drop=True)
        print(f"[debug] usando apenas {len(df)} linhas para validar o pipeline")

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    if args.mode == "split":
        train_df, val_df = group_train_val_split(df, val_size=args.val_size, seed=args.seed)
        prefix = f"{args.tag}_" if args.tag else ""
        run_name = f"{prefix}split_{int(t0)}"
        metrics, report, cm = run_one_split(train_df, val_df, args, run_name)
        log_result(run_name, args, metrics)

    else:  # kfold
        accs, f1s = [], []
        prefix = f"{args.tag}_" if args.tag else ""
        for i, (train_df, val_df) in enumerate(group_kfold_splits(df, n_splits=args.kfold, seed=args.seed)):
            run_name = f"{prefix}kfold{i}_{int(t0)}"
            print(f"\n===== FOLD {i+1}/{args.kfold} =====")
            metrics, report, cm = run_one_split(train_df, val_df, args, run_name)
            log_result(run_name, args, metrics)
            accs.append(metrics["eval_accuracy"])
            f1s.append(metrics["eval_macro_f1"])
        mean_acc, mean_f1 = float(np.mean(accs)), float(np.mean(f1s))
        print(f"\nMédia {args.kfold}-fold: acc={mean_acc:.4f} (+-{np.std(accs):.4f})  "
              f"macro-F1={mean_f1:.4f} (+-{np.std(f1s):.4f})")
        # linha de RESUMO (média dos folds) além das linhas por fold já logadas --
        # é essa que o run_overnight.py (e você, olhando o CSV) deve olhar pra
        # comparar configurações entre si, em vez de garimpar fold por fold.
        log_result(f"{prefix}kfold_SUMMARY_{int(t0)}", args,
                   {"eval_accuracy": mean_acc, "eval_macro_f1": mean_f1})

    print(f"[info] tempo total: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()