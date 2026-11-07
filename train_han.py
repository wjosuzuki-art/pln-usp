"""
train_han.py -- Hierarchical Attention Network (Yang et al., 2016) para a clareza das respostas do e-SIC.

  palavras --(BiGRU + atenção)--> vetor da frase --(BiGRU + atenção)--> vetor do documento --> 3 classes

Usa os MESMOS folds, o MESMO alvo suave para duplicatas (aggregate_duplicates) e o MESMO formato de saída
do train_v2.py (oof_probs.csv + test_probs_mean.npy), então entra direto no ensemble.py e na ablacao.py.

Embeddings: opcional, mas ajudam muito com 20 mil textos. Baixe um do repositório do NILC
(nilc.icmc.usp.br/embeddings; formato texto word2vec/fastText, ex.: skip_s300.txt ou cbow_s100.txt) e passe
--emb caminho.txt. Sem --emb, os vetores começam aleatórios (HAN "do zero").

Uso:
    python train_han.py --data train.xlsx --test test.xlsx --debug                 # teste rápido (~1 min)
    python train_han.py --data train.xlsx --folds 0 --epochs 8 --emb skip_s300.txt  # piloto: veja a curva por época
    python train_han.py --data train.xlsx --test test.xlsx --epochs 6 --emb skip_s300.txt --folds_from runs_v2/final_large_*
"""
import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from data_utils import LABELS, load_train, clean_text
from train_v2 import make_group_folds, aggregate_duplicates, metrics_from_probs


# ───────────────────────── pré-processamento (sem torch; testável em CPU) ─────────────────────────

_SENT = re.compile(r"(?<=[.!?;:])\s+")
_WORD = re.compile(r"\w+")


def split_sentences(text):
    return [s for s in _SENT.split(text) if s.strip()]


def tokenize(sent):
    return _WORD.findall(re.sub(r"\d+", "0", sent.lower()))   # números viram '0': reduz o vocabulário


def doc_tokens(text, max_sents, max_words):
    """Lista de frases, cada uma lista de palavras (corta frases longas e documentos longos)."""
    out = []
    for s in split_sentences(text):
        w = tokenize(s)
        if w:
            out.append(w[:max_words])
        if len(out) == max_sents:
            break
    return out


def build_vocab(docs, min_count):
    c = Counter(w for d in docs for s in d for w in s)
    words = [w for w, n in c.most_common() if n >= min_count]
    return {w: i + 2 for i, w in enumerate(words)}          # 0 = padding, 1 = desconhecida


def to_tensor(docs, vocab, max_sents, max_words):
    X = np.zeros((len(docs), max_sents, max_words), dtype=np.int32)
    for i, d in enumerate(docs):
        for j, s in enumerate(d):
            ids = [vocab.get(w, 1) for w in s]
            X[i, j, :len(ids)] = ids
    return X


def load_embeddings(path, vocab, dim_default):
    """Lê um arquivo de texto word2vec/fastText sem carregar tudo na memória (só as palavras do vocabulário)."""
    E = np.random.RandomState(0).normal(0, 0.1, (len(vocab) + 2, dim_default)).astype(np.float32)
    E[0] = 0
    if not path:
        return E, 0.0
    found, dim = 0, None
    with open(path, encoding="utf-8", errors="ignore") as f:
        for ln, line in enumerate(f):
            parts = line.rstrip().split(" ")
            if ln == 0 and len(parts) == 2 and all(p.isdigit() for p in parts):
                dim = int(parts[1])
                E = np.random.RandomState(0).normal(0, 0.1, (len(vocab) + 2, dim)).astype(np.float32)
                E[0] = 0
                continue
            w = parts[0].lower()
            if w in vocab and len(parts) > 2:
                vec = np.asarray(parts[1:], dtype=np.float32)
                if dim is None:
                    dim = len(vec)
                    E = np.random.RandomState(0).normal(0, 0.1, (len(vocab) + 2, dim)).astype(np.float32)
                    E[0] = 0
                if len(vec) == dim:
                    E[vocab[w]] = vec
                    found += 1
    return E, found / max(len(vocab), 1)


# ───────────────────────── modelo e treino (torch) ─────────────────────────

def build_model(torch, emb_matrix, hidden, dropout, freeze_emb):
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

    class HAN(nn.Module):
        def __init__(self):
            super().__init__()
            V, E = emb_matrix.shape
            self.emb = nn.Embedding(V, E, padding_idx=0)
            self.emb.weight.data.copy_(torch.from_numpy(emb_matrix))
            self.emb.weight.requires_grad = not freeze_emb
            self.wgru = nn.GRU(E, hidden, batch_first=True, bidirectional=True)
            self.wproj = nn.Linear(2 * hidden, 2 * hidden)
            self.wctx = nn.Parameter(torch.randn(2 * hidden) * 0.1)
            self.sgru = nn.GRU(2 * hidden, hidden, batch_first=True, bidirectional=True)
            self.sproj = nn.Linear(2 * hidden, 2 * hidden)
            self.sctx = nn.Parameter(torch.randn(2 * hidden) * 0.1)
            self.drop = nn.Dropout(dropout)
            self.fc = nn.Linear(2 * hidden, len(LABELS))

        def attend(self, h, ctx, proj, mask):
            sc = (torch.tanh(proj(h)) @ ctx).masked_fill(~mask, -1e9)
            a = torch.softmax(sc, -1) * mask
            return (a.unsqueeze(-1) * h).sum(1)

        def forward(self, x):
            B, S, W = x.shape
            wl = (x != 0).sum(-1)                                   # palavras por frase
            sm = wl > 0
            flat_idx = sm.view(-1).nonzero(as_tuple=True)[0]
            sent = torch.zeros(B * S, 2 * hidden, device=x.device)
            if len(flat_idx):
                xs = x.view(B * S, W)[flat_idx]
                ls = wl.view(-1)[flat_idx]
                e = self.drop(self.emb(xs))
                pk = pack_padded_sequence(e, ls.cpu(), batch_first=True, enforce_sorted=False)
                h, _ = pad_packed_sequence(self.wgru(pk)[0], batch_first=True, total_length=xs.shape[1])
                sv = self.attend(h, self.wctx, self.wproj, xs != 0)
                sent = sent.index_copy(0, flat_idx, sv)
            sent = sent.view(B, S, -1)
            dl = sm.sum(-1).clamp(min=1)                            # frases por documento (>=1)
            pk = pack_padded_sequence(self.drop(sent), dl.cpu(), batch_first=True, enforce_sorted=False)
            h, _ = pad_packed_sequence(self.sgru(pk)[0], batch_first=True, total_length=S)
            pos = torch.arange(S, device=x.device).unsqueeze(0) < dl.unsqueeze(1)
            doc = self.attend(h, self.sctx, self.sproj, pos)
            return self.fc(self.drop(doc))

    return HAN()


def batches(n, bs, shuffle, seed):
    idx = np.arange(n)
    if shuffle:
        np.random.RandomState(seed).shuffle(idx)
    return [idx[i:i + bs] for i in range(0, n, bs)]


def trim(xb):
    """Corta o preenchimento: usa só o nº máximo de frases e de palavras que existe no lote."""
    s = int((xb > 0).any(2).sum(1).max())
    w = int((xb > 0).sum(2).max())
    return xb[:, :max(s, 1), :max(w, 1)]


def predict(torch, model, X, device, bs=128):
    model.eval()
    out = np.zeros((len(X), len(LABELS)), dtype=np.float32)
    with torch.no_grad():
        for b in batches(len(X), bs, False, 0):
            xb = torch.from_numpy(trim(X[b]).astype(np.int64)).to(device)
            out[b] = torch.softmax(model(xb).float(), -1).cpu().numpy()
    return out


def train_fold(args, torch, emb, X_tr, soft, w, X_va, y_va, device, seed, tag):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(torch, emb, args.hidden, args.dropout, args.freeze_emb).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * int(np.ceil(len(X_tr) / args.batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1,
                                                anneal_strategy="linear")
    soft_t, w_t = torch.tensor(soft), torch.tensor(w)
    hist = []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, run, runw = time.time(), 0.0, 0.0
        for b in batches(len(X_tr), args.batch_size, True, seed * 100 + ep):
            xb = torch.from_numpy(trim(X_tr[b]).astype(np.int64)).to(device)
            logp = torch.log_softmax(model(xb).float(), -1)
            per = -(soft_t[b].to(device) * logp).sum(-1)
            wb = w_t[b].to(device)
            loss = (per * wb).sum() / wb.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += float((per.detach() * wb).sum())
            runw += float(wb.sum())
        P = predict(torch, model, X_va, device)
        m = metrics_from_probs(y_va, P)
        m.update(epoch=ep, train_loss=run / runw)
        hist.append(m)
        print(f"  [{tag}] época {ep}/{args.epochs}: loss treino={run / runw:.4f} | val acc={m['accuracy']:.4f} "
              f"log-loss={m['log_loss']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return model, P, hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="train.xlsx")
    p.add_argument("--test", default=None)
    p.add_argument("--kfold", type=int, default=5)
    p.add_argument("--folds", default=None, help="ex.: 0  ou  0,1,2  (padrão: todos)")
    p.add_argument("--folds_from", default=None,
                   help="Pasta de uma rodada anterior (usa os folds do oof_probs.csv dela; recomendado).")
    p.add_argument("--emb", default=None, help="arquivo de embeddings em texto (word2vec/fastText)")
    p.add_argument("--emb_dim", type=int, default=200, help="dimensão se NÃO houver --emb")
    p.add_argument("--freeze_emb", action="store_true")
    p.add_argument("--max_sents", type=int, default=40)
    p.add_argument("--max_words", type=int, default=64)
    p.add_argument("--min_count", type=int, default=2)
    p.add_argument("--hidden", type=int, default=100)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--dup_weight", choices=["one", "sqrt", "count"], default="sqrt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="HAN")
    p.add_argument("--out_dir", default="runs_v2")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] dispositivo: {device}")

    df = load_train(args.data)
    if args.debug:
        df = df.sample(min(600, len(df)), random_state=args.seed).reset_index(drop=True)
        args.epochs, args.kfold = 1, 3
        print(f"[debug] {len(df)} linhas, 1 época, 3 folds")
    if args.folds_from:
        import glob
        src = sorted(glob.glob(args.folds_from))[-1]
        fold_of = pd.read_csv(Path(src) / "oof_probs.csv")["fold"].values
        assert len(fold_of) == len(df), f"{src}: {len(fold_of)} linhas, esperado {len(df)} (mesmo --data?)"
        args.kfold = int(fold_of.max()) + 1
        print(f"[info] folds lidos de {src}")
    else:
        fold_of = make_group_folds(df, args.kfold)
    folds_to_run = list(range(args.kfold)) if args.folds is None else [int(x) for x in args.folds.split(",")]

    test_texts = None
    if args.test:
        test_texts = pd.read_excel(args.test)["resp_text"].apply(clean_text).tolist()

    docs = [doc_tokens(t, args.max_sents, args.max_words) for t in df["text"]]
    tdocs = [doc_tokens(t, args.max_sents, args.max_words) for t in test_texts] if test_texts else []
    n_s = np.array([len(d) for d in docs])
    print(f"[info] frases por texto: mediana {np.median(n_s):.0f}, p95 {np.percentile(n_s, 95):.0f} "
          f"(limite {args.max_sents}; {np.mean(n_s >= args.max_sents):.1%} cortados)")
    vocab = build_vocab(docs + tdocs, args.min_count)           # só texto, sem rótulos
    emb, cov = load_embeddings(args.emb, vocab, args.emb_dim)
    print(f"[info] vocabulário: {len(vocab)} palavras | dimensão {emb.shape[1]} | "
          + (f"cobertura do embedding pré-treinado: {cov:.1%}" if args.emb else "embeddings aleatórios"))
    X = to_tensor(docs, vocab, args.max_sents, args.max_words)
    Xt = to_tensor(tdocs, vocab, args.max_sents, args.max_words) if tdocs else None

    run_dir = Path(args.out_dir) / f"{args.tag}_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(run_dir / "config.json", "w"), indent=2)
    print(f"[info] saída em {run_dir}")

    y = df["label_id"].values
    pos = {t: i for i, t in enumerate(df["text"].values)}
    oof = np.full((len(df), len(LABELS)), np.nan, dtype=np.float32)
    tests, fold_metrics, t_start = [], [], time.time()
    for k in folds_to_run:
        tr_idx, va_idx = np.where(fold_of != k)[0], np.where(fold_of == k)[0]
        texts, soft, w = aggregate_duplicates(df.iloc[tr_idx], args.dup_weight, args.label_smoothing)
        X_tr = X[[pos[t] for t in texts]]                         # um exemplo por texto único
        print(f"\n===== FOLD {k + 1}/{args.kfold} | treino {len(tr_idx)} linhas -> {len(texts)} textos únicos | "
              f"validação {len(va_idx)} =====", flush=True)
        model, P, hist = train_fold(args, torch, emb, X_tr, soft, w, X[va_idx], y[va_idx], device,
                                    args.seed + k, f"fold{k}")
        oof[va_idx] = P
        fold_metrics.append(metrics_from_probs(y[va_idx], P))
        json.dump(hist, open(run_dir / f"history_fold{k}.json", "w"), indent=2)
        if Xt is not None:
            tp = predict(torch, model, Xt, device)
            tests.append(tp)
            np.save(run_dir / f"test_probs_fold{k}.npy", tp)
        del model

    covered = ~np.isnan(oof).any(1)
    overall = metrics_from_probs(y[covered], oof[covered])
    o = pd.DataFrame(oof, columns=[f"p_{l}" for l in LABELS])
    o.insert(0, "fold", fold_of)
    o.insert(0, "clarity", df["clarity"].values)
    o.to_csv(run_dir / "oof_probs.csv", index_label="row")
    json.dump({"overall_oof": overall, "folds": fold_metrics, "minutes": (time.time() - t_start) / 60,
               "rows_covered": int(covered.sum())}, open(run_dir / "summary.json", "w"), indent=2)
    if tests:
        np.save(run_dir / "test_probs_mean.npy", np.mean(tests, axis=0))
    print(f"\n=== RESULTADO HAN (out-of-fold, {int(covered.sum())} linhas) ===")
    print(f"acc={overall['accuracy']:.4f}  macro-F1={overall['macro_f1']:.4f}  log-loss={overall['log_loss']:.4f}")
    print("referências: TF-IDF+LogReg acc~0.452 | transformers ~0.455-0.459")
    print(f"tempo: {(time.time() - t_start) / 60:.1f} min | salvo em {run_dir}")


if __name__ == "__main__":
    main()
