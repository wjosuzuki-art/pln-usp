"""
triage_ptt5.py -- triagem do PTT5-v2 (T5 em português, só o encoder + média dos vetores) no FOLD 0.

Roda 4 configurações (base e large, 2 learning rates cada) e, no fim, responde a pergunta que importa:
  "isto agrega ao ensemble?"  ->  compara, nas mesmas linhas do fold 0,
    (a) acurácia sozinho vs. os modelos de referência (J2_ref, N2_ref, B2_ref, se existirem em runs_v2/);
    (b) concordância com eles (quanto mais baixa com acurácia parecida, maior o potencial);
    (c) média das referências COM e SEM o PTT5, com IC 95% pareado por texto.
Limitação honesta: o fold 0 tem ~4.200 linhas, então só efeitos grandes (> ~1 ponto) aparecem como
significativos. Use o resultado para DESCARTAR o que está claramente abaixo (< ~44%) e para escolher a
configuração a rodar em 5 folds; a decisão final é a estimativa honesta do ensemble.py.

Uso:
    python triage_ptt5.py --data train.xlsx            # SEM o test1 no treino (test1 fica como teste de verdade)
    python triage_ptt5.py --data train.xlsx --report_only
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data_utils import LABEL2ID, load_train                        # noqa: E402
from grid3f import completed_run, paired_bootstrap      # noqa: E402

BASE_M, LARGE_M = "unicamp-dl/ptt5-v2-base", "unicamp-dl/ptt5-v2-large"
COMMON = ["--kfold", "5", "--folds", "0", "--epochs", "3", "--max_len", "512", "--head_tokens", "128",
          "--llrd", "1.0", "--label_smoothing", "0.05", "--dup_weight", "sqrt", "--warmup", "0.1",
          "--weight_decay", "0.01", "--adam_beta2", "0.98", "--adam_eps", "1e-6", "--seed", "42"]
CONFIGS = [  # (tag, modelo, lr, batch, accum) -- AdamW com T5 pede lr bem maior que BERT
    ("P_base_lr1e-4",  BASE_M,  "1e-4", "16", "1"),
    ("P_base_lr3e-4",  BASE_M,  "3e-4", "16", "1"),
    ("P_large_lr1e-4", LARGE_M, "1e-4", "8", "2"),
    ("P_large_lr5e-5", LARGE_M, "5e-5", "8", "2"),
]
COLS = ["p_c1", "p_c234", "p_c5"]


def find_refs(n_rows):
    """Referências com OOF sobre ESTES dados: primeiro os modelos finais antigos (treinados em train.xlsx, 5 folds),
    senão as bases dos grids novos (treinadas em train_mais_test1.xlsx). Só vale quem tem o mesmo nº de linhas."""
    groups = [
        [("BERTimbau-final", "final_large"), ("LegalNLP-final", "final_legal"), ("NorBERTo-final", "final_g_norb_lr3e-5")],
        [("B2_ref", "B2_ref"), ("J2_ref", "J2_ref"), ("N2_ref", "N2_ref")],
    ]
    for g in groups:
        found = {}
        for label, tag in g:
            pat = re.compile(rf"^{re.escape(tag)}_\d+$")
            runs = [d for d in Path("runs_v2").glob(f"{tag}_*")
                    if pat.match(d.name) and (d / "oof_probs.csv").exists() and (d / "summary.json").exists()]
            for d in sorted(runs, key=lambda d: -d.stat().st_mtime):
                if len(pd.read_csv(d / "oof_probs.csv", usecols=["fold"])) == n_rows:
                    found[label] = d
                    break
        if len(found) >= 2:
            return found
    return {}


def load_fold0(rd):
    o = pd.read_csv(Path(rd) / "oof_probs.csv")
    return o["fold"].values, o[COLS].values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="train.xlsx")
    p.add_argument("--report_only", action="store_true")
    a = p.parse_args()
    # trava de segurança: um train_v2.py cortado na cópia faria as 4 configurações falharem em segundos
    if subprocess.run([sys.executable, "-m", "py_compile", "train_v2.py"]).returncode != 0:
        sys.exit("[erro] train_v2.py está com erro de sintaxe (cópia cortada?). Copie de novo e confira com md5sum/wc -l.")
    Path("grid_logs").mkdir(exist_ok=True)

    t_all = time.time()
    for i, (tag, model, lr, bs, ga) in enumerate(CONFIGS, 1):
        if completed_run(tag, 1):
            print(f"[{i}/{len(CONFIGS)}] {tag}: já concluída -- pulando", flush=True)
            continue
        if a.report_only:
            continue
        cmd = [sys.executable, "train_v2.py", "--data", a.data, "--tag", tag, "--model", model,
               "--lr", lr, "--batch_size", bs, "--grad_accum", ga] + COMMON
        print(f"[{i}/{len(CONFIGS)}] {tag} ... (log: grid_logs/{tag}.log)", flush=True)
        t0 = time.time()
        with open(f"grid_logs/{tag}.log", "w", encoding="utf-8") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        rd = completed_run(tag, 1)
        if rc == 0 and rd:
            m = json.load(open(rd / "summary.json"))["overall_oof"]
            print(f"    acc={m['accuracy']:.4f} log-loss={m['log_loss']:.4f} ({(time.time() - t0) / 60:.0f} min)",
                  flush=True)
        else:
            print(f"    FALHOU (código {rc}) -- veja grid_logs/{tag}.log", flush=True)

    # ------------------------------ análise no fold 0 ------------------------------
    df = load_train(a.data)
    y = df["clarity"].map(LABEL2ID).values
    texts = df["text"].values
    ptt = {t: completed_run(t, 1) for t, *_ in CONFIGS}
    ptt = {t: d for t, d in ptt.items() if d}
    refs = find_refs(len(df))
    if not refs:
        print(f"[aviso] nenhuma referência com OOF sobre {len(df)} linhas em runs_v2/ -- sem comparação com o ensemble.")
    if not ptt:
        print("(nenhuma configuração do PTT5 concluída)")
        return
    fold, _ = load_fold0(next(iter(ptt.values())))
    m0 = fold == 0
    ym, tm = y[m0], texts[m0]
    acc = lambda P: float((P.argmax(1) == ym).mean())
    ll = lambda P: float(-np.log(np.clip(P[np.arange(len(ym)), ym], 1e-7, 1)).mean())
    cor = lambda P: P.argmax(1) == ym

    R = {}
    for t, d in refs.items():
        f, P = load_fold0(d)
        if len(f) == len(df) and np.array_equal(f, fold):
            R[t] = P[m0]
        else:
            print(f"[aviso] {t}: divisão de folds diferente da do PTT5 -- ignorada")
    lines = [f"=== TRIAGEM PTT5-v2 no fold 0 ({int(m0.sum())} linhas) | dados: {a.data} ({len(df)} linhas) ===",
             "Erro-padrão de uma acurácia ~46% com esse tamanho: ~0,8 ponto.\n",
             f"{'modelo':16s} {'acc':>7s} {'log-loss':>9s}   concordância com " + " / ".join(R)]
    for t, P in R.items():
        lines.append(f"{t:16s} {acc(P) * 100:6.2f}% {ll(P):9.4f}   (referência)")
    best = None
    for t, d in ptt.items():
        _, P = load_fold0(d)
        P = P[m0]
        ag = " / ".join(f"{(P.argmax(1) == Q.argmax(1)).mean() * 100:4.1f}%" for Q in R.values())
        lines.append(f"{t:16s} {acc(P) * 100:6.2f}% {ll(P):9.4f}   {ag}")
        if best is None or acc(P) > acc(best[1]):
            best = (t, P)
    if len(R) >= 2 and best:
        base = np.mean(list(R.values()), axis=0)
        lines.append(f"\n=== MARGINAL: média simples das {len(R)} referências, com e sem o PTT5 (fold 0, sem TF-IDF) ===")
        lines.append(f"  sem PTT5:                    acc={acc(base) * 100:6.2f}%  log-loss={ll(base):.4f}")
        for t, d in ptt.items():
            _, P = load_fold0(d)
            P = P[m0]
            w = np.mean(list(R.values()) + [P], axis=0)
            dlt = acc(w) - acc(base)
            lo, hi = paired_bootstrap(cor(w), cor(base), tm)
            lines.append(f"  com {t:16s}      acc={acc(w) * 100:6.2f}%  log-loss={ll(w):.4f}   "
                         f"{dlt * 100:+.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}]{' *' if lo > 0 else ''}")
        lines.append("\nComo ler: PTT5 sozinho >= ~44,5% E concordância com as referências mais baixa que a delas entre si")
        lines.append("(61,9% a 71,0%) = candidato. Diferença no marginal dentro do IC = inconclusivo (fold único).")
    if best:
        t = best[0]
        cfg = next(c for c in CONFIGS if c[0] == t)
        lines.append(f"\nMelhor sozinho: {t}. Para treinar em 5 folds (gera OOF completo + previsões do teste):")
        lines.append(f"  python train_v2.py --data {a.data} --test test.xlsx --tag P_final_{t[2:]} --model {cfg[1]} "
                     f"--lr {cfg[2]} --batch_size {cfg[3]} --grad_accum {cfg[4]} --kfold 5 "
                     + " ".join(x for x in COMMON[4:]))
    text = "\n".join(lines)
    Path("leaderboard_ptt5.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"\n({(time.time() - t_all) / 60:.0f} min nesta execução)")


if __name__ == "__main__":
    main()
