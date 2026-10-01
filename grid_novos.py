"""
grid_novos.py -- grid dos modelos novos, pensado para rodar sozinho numa
RTX 3060 (12GB) durante ~12-14h.

Decisões (baseadas no que já medimos):
  * Albertina-900M: LoRA (base congelada em bf16). Fine-tuning completo
    precisa de ~14GB só de pesos+gradientes+otimizador -> não cabe.
    Com LoRA, learning rate bem maior (1e-4 a 2e-4) e sem LLRD.
  * NorBERTo-large (395M, ModernBERT): fine-tuning completo. ModernBERT
    costuma preferir lr um pouco maior que o BERT clássico.
  * Uma configuração com 1024 tokens: o NorBERTo aceita textos longos,
    e 1024 tokens cobrem praticamente todas as respostas inteiras.
  * Ordem = prioridade, alternando os modelos: se precisar interromper,
    o mais promissor de cada um já rodou.

Todas as configurações: 1 fold (fold 0 de 5), 3 épocas, sem salvar modelo.

Com --auto_final, ao terminar o grid ele roda o vencedor com os 5 folds
e --save_models -- mas SÓ se o tempo estimado couber em --budget_h
(estimativa = 5 x o tempo que o vencedor levou no grid).

Uso:
    python grid_novos.py --data train.xlsx --auto_final --budget_h 13
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ALB = "PORTULAN/albertina-900m-portuguese-ptbr-encoder"
NORB = "Itau-Unibanco/NorBERTo-large"
LORA = ["--lora", "--llrd", "1.0"]

# tempo estimado por item na RTX 3060 (grosseiro; o log mostra o real)
GRID = [
    {"tag": "g_norb_lr3e-5", "model": NORB,          # ~40-50 min
     "args": ["--lr", "3e-5", "--batch_size", "8", "--grad_accum", "2"]},
    {"tag": "g_alb_lora16_lr2e-4", "model": ALB,     # ~60-90 min
     "args": LORA + ["--lr", "2e-4", "--lora_r", "16", "--lora_alpha", "32",
                     "--batch_size", "4", "--grad_accum", "4"]},
    {"tag": "g_norb_lr5e-5", "model": NORB,          # ~40-50 min
     "args": ["--lr", "5e-5", "--batch_size", "8", "--grad_accum", "2"]},
    {"tag": "g_alb_lora16_lr1e-4", "model": ALB,     # ~60-90 min
     "args": LORA + ["--lr", "1e-4", "--lora_r", "16", "--lora_alpha", "32",
                     "--batch_size", "4", "--grad_accum", "4"]},
    {"tag": "g_norb_lr3e-5_len1024", "model": NORB,  # ~60-80 min
     "args": ["--lr", "3e-5", "--max_len", "1024", "--batch_size", "4", "--grad_accum", "4"]},
    {"tag": "g_norb_lr2e-5", "model": NORB,          # ~40-50 min
     "args": ["--lr", "2e-5", "--batch_size", "8", "--grad_accum", "2"]},
    {"tag": "g_alb_lora32_lr2e-4", "model": ALB,     # ~60-90 min
     "args": LORA + ["--lr", "2e-4", "--lora_r", "32", "--lora_alpha", "64",
                     "--batch_size", "4", "--grad_accum", "4"]},
]


def run(cmd, log):
    with open(log, "w", encoding="utf-8") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="train.xlsx")
    p.add_argument("--kfold", type=int, default=5)
    p.add_argument("--fold", default="0")
    p.add_argument("--auto_final", action="store_true",
                   help="Depois do grid, roda o vencedor com 5 folds + --save_models (se couber no tempo).")
    p.add_argument("--budget_h", type=float, default=13.0,
                   help="Horas totais disponíveis (grid + final).")
    a = p.parse_args()

    logs = Path("grid_logs")
    logs.mkdir(exist_ok=True)
    results = []
    t_all = time.time()

    for i, g in enumerate(GRID, 1):
        cmd = [sys.executable, "train_v2.py", "--data", a.data, "--kfold", str(a.kfold),
               "--folds", a.fold, "--tag", g["tag"], "--model", g["model"]] + g["args"]
        log = logs / f"{g['tag']}.log"
        print(f"[{i}/{len(GRID)}] {g['tag']} ... (log: {log})", flush=True)
        t0 = time.time()
        rc = run(cmd, log)
        mins = (time.time() - t0) / 60
        runs = sorted(Path("runs_v2").glob(f"{g['tag']}_*"), key=lambda d: d.stat().st_mtime)
        summ = runs[-1] / "summary.json" if runs else None
        if rc == 0 and summ and summ.exists():
            m = json.load(open(summ))["overall_oof"]
            results.append((g, m, mins))
            print(f"    acc={m['accuracy']:.4f} macro-F1={m['macro_f1']:.4f} "
                  f"log-loss={m['log_loss']:.4f} ({mins:.0f} min)", flush=True)
        else:
            results.append((g, None, mins))
            print(f"    FALHOU (código {rc}) -- veja {log}. Seguindo.", flush=True)

    ok = sorted([r for r in results if r[1]], key=lambda r: (-r[1]["accuracy"], r[1]["log_loss"]))
    elapsed_h = (time.time() - t_all) / 3600
    lines = [f"=== GRID NOVOS MODELOS (fold {a.fold} de {a.kfold}) -- {elapsed_h:.1f}h ===",
             "Erro-padrão da acurácia num fold só: ~0,8 ponto.",
             "Referências (fold 0 do grid anterior): BERTimbau-large 0,4591 | base 0,4578 | jurídico 0,4568",
             "Referências (5 folds): BERTimbau-large 0,4554 | jurídico 0,4558 | ensemble 3 modelos 0,4726\n"]
    for g, m, mins in ok:
        lines.append(f"{g['tag']:24s} acc={m['accuracy']:.4f} macro-F1={m['macro_f1']:.4f} "
                     f"log-loss={m['log_loss']:.4f}  ({mins:.0f} min)")
    for g, m, mins in results:
        if m is None:
            lines.append(f"{g['tag']:24s} FALHOU")

    final_cmd = None
    if ok:
        best, _, best_min = ok[0]
        final_cmd = [sys.executable, "train_v2.py", "--data", a.data, "--kfold", str(a.kfold),
                     "--save_models", "--tag", f"final_{best['tag']}", "--model", best["model"]] + best["args"]
        lines.append(f"\nVencedor: {best['tag']}. Comando dos 5 folds finais:")
        lines.append(" ".join(final_cmd[1:]).replace(sys.executable, "python"))
    text = "\n".join(lines)
    Path("leaderboard_novos.txt").write_text(text, encoding="utf-8")
    print("\n" + text, flush=True)

    if a.auto_final and final_cmd:
        est_h = 5 * best_min / 60
        if elapsed_h + est_h <= a.budget_h:
            print(f"\n[auto_final] rodando os 5 folds de {best['tag']} "
                  f"(estimativa ~{est_h:.1f}h; total previsto {elapsed_h + est_h:.1f}h de {a.budget_h}h)", flush=True)
            rc = run(final_cmd, logs / f"final_{best['tag']}.log")
            print(f"[auto_final] terminou com código {rc} -- log em grid_logs/final_{best['tag']}.log", flush=True)
        else:
            print(f"\n[auto_final] PULADO: estimativa de {est_h:.1f}h não cabe no orçamento "
                  f"({elapsed_h:.1f}h já usadas de {a.budget_h}h). Rode o comando acima manualmente.", flush=True)


if __name__ == "__main__":
    main()