"""
train_v2.py -- versão 2 do pipeline, pensada para a RTX 3060 Ti (8GB, Ampere).

O que muda em relação ao train_transformer.py (e por quê):

1. MODELO MAIOR: padrão = BERTimbau-large (335M parâmetros, 3x o base).
   Em benchmarks de classificação em português (tabelas do paper do
   NorBERTo, PROPOR 2026), o large supera o base de forma consistente.
   Alternativas testáveis via --model:
     - FacebookAI/xlm-roberta-large  (560M; o melhor em uma das tarefas de
       classificação daquele benchmark, mas é pesado e às vezes instável)
     - Itau-Unibanco/NorBERTo-large  (ModernBERT, 2026, contexto longo --
       o candidato "mais novo"; confira o nome exato no Hugging Face)

2. TRUNCAMENTO HEAD+TAIL: em vez de cortar o texto no token 256/512 e
   jogar fora o fim, guardamos o COMEÇO e o FIM (padrão: 128 tokens
   iniciais + o restante do orçamento no final). Em cartas do e-SIC o
   fim costuma ter o "miolo" da resposta (links, prazos, recursos).
   É a estratégia que se saiu melhor no estudo clássico de Sun et al.
   (2019), "How to Fine-Tune BERT for Text Classification?".

3. DUPLICATAS COMO DISTRIBUIÇÃO DE RÓTULO: cada texto idêntico vira UMA
   linha de treino com um alvo "suave" (ex.: 3 cópias com c1,c1,c234 ->
   alvo [0.67, 0.33, 0]) e um peso controlado por --dup_weight. Com
   peso = nº de cópias isso é matematicamente igual a treinar com as
   linhas repetidas; com peso = raiz (padrão) ou 1, as cartas-modelo
   deixam de dominar o treino só por serem repetidas. Label smoothing
   leve completa a defesa contra o ruído de rótulo que medimos.

4. LOOP DE TREINO PRÓPRIO (sem o Trainer): controle total, loss exibido
   na escala certa (acabou o "16,xx"), bf16 automático em Ampere,
   batches agrupados por tamanho (bem mais rápido), layer-wise LR decay
   (camadas de baixo mudam menos -- estabiliza modelos large) e uma
   trava contra "colapso" (large às vezes degenera num fold; o script
   detecta e reinicia aquele fold com outra seed).

5. TODOS OS FOLDS VIRAM O MODELO FINAL: cada fold prevê o teste e as
   previsões são MÉDIAS entre os folds (ensemble). Nada de escolher "o
   checkpoint do fold X". Também salva as previsões out-of-fold (OOF)
   para o ensemble.py combinar modelos diferentes depois.

6. RESGATE DE DUPLICATAS NO TESTE: se um texto do teste é idêntico a
   algum do treino, as probabilidades do modelo são combinadas com os
   rótulos que aquele texto recebeu no treino (atualização bayesiana
   simples). Base empírica: "copiar a maioria das outras cópias"
   acerta ~49,8% nas duplicatas (validado leave-one-out). Sai em um
   arquivo separado para vocês escolherem (e declararem no relatório).

Uso:
    # teste rápido (poucos minutos) -- SEMPRE rode antes:
    python train_v2.py --data train.xlsx --test test.xlsx --debug

    # rodada completa:
    python train_v2.py --data train.xlsx --test test.xlsx

    # outro modelo, com tag pra identificar no results_log.csv:
    python train_v2.py --data train.xlsx --test test.xlsx \
        --model FacebookAI/xlm-roberta-large --lr 8e-6 --tag xlmr

--test é opcional; sem ele, só faz a validação cruzada.
"""

import argparse
import csv
import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold

from data_utils import LABELS, LABEL2ID, ID2LABEL, load_train, clean_text

RESULTS_LOG = "results_log.csv"
N_CLASSES = len(LABELS)


# ─────────────────────────── argumentos ────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="train.xlsx")
    p.add_argument("--test", default=None, help="test.xlsx sem rótulos (opcional)")
    p.add_argument("--model", default="neuralmind/bert-large-portuguese-cased")
    p.add_argument("--kfold", type=int, default=3)
    p.add_argument("--folds", default=None,
                   help="Rodar só alguns folds, ex.: '0' ou '0,1' (útil pra testes). Padrão: todos.")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--head_tokens", type=int, default=128,
                   help="Tokens mantidos do INÍCIO do texto; o resto do orçamento vai pro FIM.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1.5e-5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--llrd", type=float, default=0.9,
                   help="Layer-wise LR decay (1.0 = desliga).")
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--dup_weight", choices=["one", "sqrt", "count"], default="sqrt",
                   help="Peso de um texto repetido n vezes: 1, sqrt(n) ou n.")
    p.add_argument("--dup_blend_alpha", type=float, default=3.0,
                   help="Força do modelo vs. rótulos de treino no resgate de duplicatas do teste "
                        "(maior = confia mais no modelo).")
    p.add_argument("--no_grad_ckpt", action="store_true",
                   help="Desliga gradient checkpointing (mais rápido, mais VRAM).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--optim_8bit", action="store_true",
                   help="Usa AdamW de 8 bits (biblioteca bitsandbytes) em vez do AdamW padrão. "
                        "Corta a memória do estado do otimizador em ~4x -- necessário para "
                        "modelos grandes (ex.: Albertina-900M) numa GPU de 12GB. "
                        "Requer: pip install bitsandbytes")
    p.add_argument("--lora", action="store_true",
                   help="Treina com LoRA: base congelada em bf16 + adaptadores pequenos. "
                        "Necessário para o Albertina-900M numa GPU de 12GB. Requer: pip install peft. "
                        "Com LoRA use lr bem maior (1e-4 a 3e-4) e --llrd 1.0.")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--debug", action="store_true",
                   help="Amostra pequena, 1 época, max_len 128: só pra validar o pipeline.")
    p.add_argument("--out_dir", default="runs_v2")
    p.add_argument("--tag", default="")
    p.add_argument("--save_models", action="store_true",
                   help="Salva o modelo de cada fold em <run_dir>/model_foldK, para prever o teste "
                        "depois com predict_v2.py sem treinar de novo. Ocupa disco: ~1,3GB por fold "
                        "no BERTimbau-large (~0,45GB no base).")
    return p.parse_args()


# ─────────────────────── utilitários sem torch ─────────────────────
# (separados para poderem ser testados sem GPU)

def make_group_folds(df, n_splits):
    """Mesma lógica de antes: GroupKFold agrupado pelo texto exato.
    Determinístico (sem shuffle) -> o ensemble.py reconstrói os mesmos folds."""
    fold = np.full(len(df), -1, dtype=int)
    for k, (_, va) in enumerate(GroupKFold(n_splits=n_splits).split(df, groups=df["text"])):
        fold[va] = k
    return fold


def aggregate_duplicates(train_df, dup_weight, label_smoothing):
    """Colapsa textos idênticos em 1 linha com alvo suave + peso."""
    counts = (train_df.groupby(["text", "label_id"]).size()
              .unstack(fill_value=0)
              .reindex(columns=range(N_CLASSES), fill_value=0))
    n = counts.sum(axis=1).values.astype(float)
    soft = counts.values / n[:, None]
    if label_smoothing > 0:
        soft = (1 - label_smoothing) * soft + label_smoothing / N_CLASSES
    if dup_weight == "one":
        w = np.ones_like(n)
    elif dup_weight == "sqrt":
        w = np.sqrt(n)
    else:
        w = n
    return counts.index.tolist(), soft.astype(np.float32), w.astype(np.float32)


def get_special_wrapping(tokenizer):
    """Descobre o prefixo/sufixo de tokens especiais QUE O TOKENIZER REALMENTE
    ADICIONA, comparando um texto de prova com e sem add_special_tokens --
    em vez de assumir nomes de atributo (cls_token_id, sep_token_id, ...) ou
    métodos (build_inputs_with_special_tokens) que variam entre famílias de
    tokenizer (BERT, DeBERTa-v2, ModernBERT, ...) e às vezes nem existem."""
    probe = tokenizer("x", add_special_tokens=False)["input_ids"]
    full = tokenizer("x", add_special_tokens=True)["input_ids"]
    for i in range(len(full) - len(probe) + 1):
        if full[i:i + len(probe)] == probe:
            return full[:i], full[i + len(probe):]
    raise ValueError("Não consegui identificar os tokens especiais deste tokenizer.")


def encode_head_tail(tokenizer, texts, max_len, head_tokens):
    """Tokeniza guardando o início e o fim de textos longos."""
    prefix, suffix = get_special_wrapping(tokenizer)
    budget = max_len - len(prefix) - len(suffix)
    raw = tokenizer(list(texts), add_special_tokens=False, truncation=False)["input_ids"]
    out = []
    for ids in raw:
        if len(ids) > budget:
            h = min(head_tokens, budget)
            t = budget - h
            ids = ids[:h] + (ids[-t:] if t > 0 else [])
        out.append(prefix + ids + suffix)
    return out


def length_bucketed_batches(lengths, batch_size, shuffle, seed):
    """Batches com textos de tamanho parecido -> muito menos padding.
    No treino: embaralha, agrupa em blocos de 50 batches, ordena dentro
    do bloco e embaralha a ordem dos batches (mantém aleatoriedade)."""
    lengths = np.asarray(lengths)
    if not shuffle:
        order = np.argsort(lengths)
        return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    rng = random.Random(seed)
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    idx = np.array(idx)
    chunk = batch_size * 50
    batches = []
    for s in range(0, len(idx), chunk):
        c = idx[s:s + chunk]
        c = c[np.argsort(lengths[c], kind="stable")]
        batches += [c[j:j + batch_size] for j in range(0, len(c), batch_size)]
    rng.shuffle(batches)
    return batches


def dup_blend(test_texts, train_df, probs, alpha):
    """Combina as probabilidades do modelo com os rótulos que textos
    IDÊNTICOS receberam no treino: p' = (alpha*p + contagens)/(alpha + n)."""
    counts = (train_df.groupby(["text", "label_id"]).size()
              .unstack(fill_value=0)
              .reindex(columns=range(N_CLASSES), fill_value=0))
    lookup = {t: row.values.astype(float) for t, row in counts.iterrows()}
    out = probs.copy()
    n_hit = 0
    for i, t in enumerate(test_texts):
        c = lookup.get(t)
        if c is not None:
            out[i] = (alpha * probs[i] + c) / (alpha + c.sum())
            n_hit += 1
    return out, n_hit


def metrics_from_probs(y_true, probs):
    probs = probs / probs.sum(1, keepdims=True)
    pred = probs.argmax(1)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "log_loss": float(log_loss(y_true, np.clip(probs, 1e-7, 1), labels=list(range(N_CLASSES)))),
    }


def log_result(run_name, args, m):
    header = ["run_name", "model", "mode", "lr", "epochs", "batch_size", "grad_accum",
              "ordinal_loss", "accuracy", "macro_f1"]
    row = [run_name, f"v2:{args.model}", "kfold", args.lr, args.epochs, args.batch_size,
           args.grad_accum, False, m["accuracy"], m["macro_f1"]]
    new = not Path(RESULTS_LOG).exists()
    with open(RESULTS_LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def write_submission(test_path, probs, out_path):
    """Mantém o arquivo de teste intacto (linhas, ordem, colunas) e só
    preenche/substitui a coluna 'clarity'."""
    tdf = pd.read_excel(test_path)
    n_in = len(tdf)
    tdf["clarity"] = [ID2LABEL[int(i)] for i in probs.argmax(1)]
    assert len(tdf) == n_in and tdf["clarity"].notna().all()
    tdf.to_excel(out_path, index=False)
    return tdf["clarity"].value_counts(normalize=True).round(3).to_dict()


# ───────────────────────── parte com torch ─────────────────────────

def setup_precision(torch):
    if not torch.cuda.is_available():
        print("[aviso] CUDA não disponível -- vai rodar em CPU (MUITO lento).")
        return "cpu", None
    major, _ = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    if major >= 8 and torch.cuda.is_bf16_supported():
        print(f"[info] {name}: usando bf16 (Ampere+).")
        return "cuda", torch.bfloat16
    if major >= 7:
        print(f"[info] {name}: usando fp16 com GradScaler.")
        return "cuda", torch.float16
    print(f"[info] {name}: Pascal ou anterior -> fp32.")
    return "cuda", None


def param_groups(model, lr, wd, decay):
    """Layer-wise LR decay: camada do topo recebe lr, a de baixo lr*decay^k."""
    pat = re.compile(r"\.layers?\.(\d+)\.")
    layer_ids = [int(m.group(1)) for n, _ in model.named_parameters() if (m := pat.search(n))]
    n_layers = max(layer_ids) + 1 if layer_ids else 0
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        m = pat.search(n)
        if m:
            depth = int(m.group(1)) + 1
        elif "embed" in n:
            depth = 0
        else:
            depth = n_layers + 1  # cabeça de classificação, pooler, norma final
        scale = decay ** (n_layers + 1 - depth)
        no_decay = p.ndim == 1 or n.endswith(".bias") or "norm" in n.lower()
        key = (depth, no_decay)
        if key not in groups:
            groups[key] = {"params": [], "lr": lr * scale,
                           "weight_decay": 0.0 if no_decay else wd}
        groups[key]["params"].append(p)
    return list(groups.values())


def load_model(args, torch):
    from transformers import AutoModelForSequenceClassification
    kw = dict(num_labels=N_CLASSES, id2label=ID2LABEL, label2id=LABEL2ID)
    if args.lora:
        # base congelada em bf16: metade da memória, e ela não recebe atualização
        kw["dtype"] = torch.bfloat16

    def _load(**extra):
        try:
            return AutoModelForSequenceClassification.from_pretrained(args.model, **kw, **extra)
        except TypeError:  # versões antigas do transformers chamam 'dtype' de 'torch_dtype'
            if "dtype" in kw:
                kw["torch_dtype"] = kw.pop("dtype")
            return AutoModelForSequenceClassification.from_pretrained(args.model, **kw, **extra)

    try:
        model = _load(use_safetensors=True)
    except Exception as e:
        print(f"[info] sem safetensors para {args.model} ({type(e).__name__}); tentando formato .bin")
        model = _load()

    if not args.no_grad_ckpt:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        if args.lora and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    if args.lora:
        model = apply_lora(model, args, torch)
    return model


def apply_lora(model, args, torch):
    """LoRA genérico: adaptadores em TODAS as camadas lineares dentro dos blocos
    do encoder (atenção + MLP), detectadas pelo nome ('.layer.N.' no BERT/DeBERTa,
    '.layers.N.' no ModernBERT) -- sem depender dos nomes internos de cada
    arquitetura. A cabeça de classificação (e o pooler, se houver) é treinada
    por inteiro, porque começa do zero."""
    import torch.nn as nn
    from peft import LoraConfig, get_peft_model

    pat = re.compile(r"\.layers?\.\d+\.")
    targets = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and pat.search(n)]
    assert targets, "Não encontrei camadas lineares dentro dos blocos do encoder para aplicar LoRA."
    prefix = getattr(model, "base_model_prefix", "")
    heads = [n for n, c in model.named_children()
             if n != prefix and any(True for _ in c.parameters())]
    cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                     target_modules=targets, modules_to_save=heads, bias="none")
    model = get_peft_model(model, cfg)
    # adaptadores e cabeça em fp32: atualizações pequenas não "somem" como em bf16
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[lora] {len(targets)} camadas adaptadas | cabeça treinada por inteiro: {heads} | "
          f"treináveis: {n_train/1e6:.1f}M de {n_all/1e6:.0f}M ({n_train/n_all:.2%})", flush=True)
    return model


def collate(torch, seqs, pad_id):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        att[i, :len(s)] = 1
    return ids, att


def predict_probs(torch, model, seqs, pad_id, bs, device, amp_dtype):
    model.eval()
    probs = np.zeros((len(seqs), N_CLASSES), dtype=np.float32)
    lengths = [len(s) for s in seqs]
    with torch.no_grad():
        for b in length_bucketed_batches(lengths, bs, shuffle=False, seed=0):
            ids, att = collate(torch, [seqs[i] for i in b], pad_id)
            ids, att = ids.to(device), att.to(device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(input_ids=ids, attention_mask=att).logits
            probs[b] = torch.softmax(logits.float(), -1).cpu().numpy()
    return probs


def train_fold(args, torch, tokenizer, train_seqs, soft, weights, val_seqs, y_val,
               device, amp_dtype, seed, fold_tag):
    from transformers import get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = load_model(args, torch).to(device)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    groups = param_groups(model, args.lr, args.weight_decay, args.llrd)
    if args.optim_8bit:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(groups, lr=args.lr, betas=(0.9, 0.98), eps=1e-6)
    else:
        opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.98), eps=1e-6)
    lengths = [len(s) for s in train_seqs]
    steps_per_epoch = math.ceil(math.ceil(len(train_seqs) / args.batch_size) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 else None

    soft_t = torch.tensor(soft)
    w_t = torch.tensor(weights)
    history = []
    val_probs = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        batches = length_bucketed_batches(lengths, args.batch_size, shuffle=True, seed=seed * 100 + epoch)
        run_loss, run_w, t0 = 0.0, 0.0, time.time()
        opt.zero_grad(set_to_none=True)
        for bi, b in enumerate(batches, 1):
            ids, att = collate(torch, [train_seqs[i] for i in b], pad_id)
            ids, att = ids.to(device), att.to(device)
            tgt = soft_t[b].to(device)
            w = w_t[b].to(device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(input_ids=ids, attention_mask=att).logits
            logp = torch.log_softmax(logits.float(), -1)
            per_ex = -(tgt * logp).sum(-1)
            loss = (per_ex * w).sum() / w.sum()
            (scaler.scale(loss / args.grad_accum) if scaler else loss / args.grad_accum).backward()
            run_loss += float((per_ex.detach() * w).sum())
            run_w += float(w.sum())

            if bi % args.grad_accum == 0 or bi == len(batches):
                if scaler:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)

            if bi % 200 == 0 or bi == len(batches):
                el = time.time() - t0
                eta = el / bi * (len(batches) - bi)
                print(f"  [{fold_tag}] época {epoch} {bi}/{len(batches)} "
                      f"loss={run_loss / run_w:.4f} ({el / 60:.1f} min, faltam ~{eta / 60:.1f} min)", flush=True)

        val_probs = predict_probs(torch, model, val_seqs, pad_id, args.eval_batch_size, device, amp_dtype)
        m = metrics_from_probs(y_val, val_probs)
        m["epoch"] = epoch
        m["train_loss"] = run_loss / run_w
        history.append(m)
        print(f"  [{fold_tag}] fim da época {epoch}: val acc={m['accuracy']:.4f} "
              f"macro-F1={m['macro_f1']:.4f} log-loss={m['log_loss']:.4f}", flush=True)

        # trava anti-colapso: large às vezes degenera prevendo 1 classe só
        if epoch == 1 and m["macro_f1"] < 0.25:
            return None, history, None

    return model, history, val_probs


def main():
    args = parse_args()
    import torch
    from transformers import AutoTokenizer
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()

    if args.debug:
        args.epochs, args.max_len = 1, 128
        args.head_tokens = min(args.head_tokens, 48)

    df = load_train(args.data)
    if args.debug:
        df = df.sample(min(600, len(df)), random_state=args.seed).reset_index(drop=True)
        print(f"[debug] {len(df)} linhas, 1 época, max_len {args.max_len}")

    device, amp_dtype = setup_precision(torch)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    fold_of = make_group_folds(df, args.kfold)
    folds_to_run = (list(range(args.kfold)) if args.folds is None
                    else [int(x) for x in args.folds.split(",")])

    stamp = int(time.time())
    run_dir = Path(args.out_dir) / f"{(args.tag + '_') if args.tag else ''}{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(run_dir / "config.json", "w"), indent=2)
    print(f"[info] saída em {run_dir}")

    test_seqs, test_texts = None, None
    if args.test:
        tdf = pd.read_excel(args.test)
        assert "resp_text" in tdf.columns, f"test sem coluna resp_text: {tdf.columns.tolist()}"
        test_texts = tdf["resp_text"].apply(clean_text).tolist()
        test_seqs = encode_head_tail(tokenizer, test_texts, args.max_len, args.head_tokens)
        overlap = sum(t in set(df["text"]) for t in test_texts)
        print(f"[info] teste: {len(test_texts)} linhas, {overlap} idênticas a algum texto do treino "
              f"({overlap / len(test_texts):.1%}) -> candidatas ao resgate de duplicatas")

    y_all = df["label_id"].values
    oof = np.full((len(df), N_CLASSES), np.nan, dtype=np.float32)
    test_probs_folds = []
    fold_metrics = []
    t_start = time.time()

    for k in folds_to_run:
        tr_df = df[fold_of != k]
        va_idx = np.where(fold_of == k)[0]
        va_df = df.iloc[va_idx]
        texts, soft, weights = aggregate_duplicates(tr_df, args.dup_weight, args.label_smoothing)
        train_seqs = encode_head_tail(tokenizer, texts, args.max_len, args.head_tokens)
        val_seqs = encode_head_tail(tokenizer, va_df["text"].tolist(), args.max_len, args.head_tokens)
        print(f"\n===== FOLD {k + 1}/{args.kfold} | treino: {len(tr_df)} linhas -> {len(texts)} textos "
              f"únicos | validação: {len(va_df)} linhas =====", flush=True)

        model, history, val_probs = None, None, None
        for attempt, seed in enumerate([args.seed + k, args.seed + k + 1000]):
            model, history, val_probs = train_fold(
                args, torch, tokenizer, train_seqs, soft, weights, val_seqs, va_df["label_id"].values,
                device, amp_dtype, seed, fold_tag=f"fold{k}")
            if model is not None:
                break
            print(f"  [fold{k}] colapso detectado (macro-F1 < 0.25 na época 1) -- "
                  f"reiniciando com outra seed...", flush=True)
            torch.cuda.empty_cache()
        if model is None:
            print(f"  [fold{k}] colapsou duas vezes; tente --lr menor. Pulando fold.")
            continue

        oof[va_idx] = val_probs
        m = metrics_from_probs(va_df["label_id"].values, val_probs)
        fold_metrics.append(m)
        log_result(f"{args.tag + '_' if args.tag else ''}v2_fold{k}_{stamp}", args, m)
        json.dump(history, open(run_dir / f"history_fold{k}.json", "w"), indent=2)

        if args.save_models:
            mdir = run_dir / f"model_fold{k}"
            if args.lora:
                # incorpora os adaptadores aos pesos -> vira um modelo comum, que o
                # predict_v2.py carrega sem precisar da biblioteca peft
                model = model.merge_and_unload()
            model.save_pretrained(mdir)
            tokenizer.save_pretrained(mdir)
            print(f"  [fold{k}] modelo salvo em {mdir}", flush=True)

        if test_seqs is not None:
            tp = predict_probs(torch, model, test_seqs, pad_id, args.eval_batch_size, device, amp_dtype)
            test_probs_folds.append(tp)
            np.save(run_dir / f"test_probs_fold{k}.npy", tp)

        del model
        torch.cuda.empty_cache()

    # ───────────── resumo, OOF e previsões finais ─────────────
    covered = ~np.isnan(oof).any(1)
    if covered.sum() == 0:
        print("[erro] nenhum fold terminou.")
        return
    overall = metrics_from_probs(y_all[covered], oof[covered])
    log_result(f"{args.tag + '_' if args.tag else ''}v2_kfold_SUMMARY_{stamp}", args, overall)

    oof_df = pd.DataFrame(oof, columns=[f"p_{l}" for l in LABELS])
    oof_df.insert(0, "fold", fold_of)
    oof_df.insert(0, "clarity", df["clarity"].values)
    oof_df.to_csv(run_dir / "oof_probs.csv", index_label="row")

    summary = {"overall_oof": overall, "folds": fold_metrics,
               "minutes": (time.time() - t_start) / 60, "rows_covered": int(covered.sum())}

    if test_probs_folds:
        mean_probs = np.mean(test_probs_folds, axis=0)
        np.save(run_dir / "test_probs_mean.npy", mean_probs)
        dist_a = write_submission(args.test, mean_probs, run_dir / "submission_model.xlsx")
        blended, n_hit = dup_blend(test_texts, df, mean_probs, args.dup_blend_alpha)
        np.save(run_dir / "test_probs_dupblend.npy", blended)
        dist_b = write_submission(args.test, blended, run_dir / "submission_dupblend.xlsx")
        changed = int((blended.argmax(1) != mean_probs.argmax(1)).sum())
        summary["test"] = {"dist_model": dist_a, "dist_dupblend": dist_b,
                           "dup_hits": n_hit, "labels_changed_by_blend": changed}
        print(f"\n[teste] submission_model.xlsx -> {dist_a}")
        print(f"[teste] submission_dupblend.xlsx -> {dist_b} "
              f"({n_hit} textos com cópia no treino, {changed} rótulos mudaram)")

    json.dump(summary, open(run_dir / "summary.json", "w"), indent=2)
    print(f"\n=== RESULTADO (out-of-fold, {covered.sum()} linhas) ===")
    print(f"acc={overall['accuracy']:.4f}  macro-F1={overall['macro_f1']:.4f}  "
          f"log-loss={overall['log_loss']:.4f}")
    print(f"referências: baseline TF-IDF+LogReg acc=0.456 / log-loss=1.044 | "
          f"BERTimbau-base (v1, fold 1, época 2) acc=0.454")
    print(f"tempo total: {summary['minutes']:.1f} min | tudo salvo em {run_dir}")


if __name__ == "__main__":
    main()