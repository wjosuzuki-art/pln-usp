"""
grid_v2.py -- grid search para o train_v2.py.

Estratégia (padrão para modelos caros):
  1. cada configuração é avaliada em UM fold (--folds 0 de um kfold=5,
     ou seja, treina com 80% e valida com 20% do train.xlsx);
  2. NENHUM modelo é salvo -- só métricas, no results_log.csv;
  3. no fim, sai um leaderboard_v2.txt ordenado por acurácia;
  4. o vencedor é retreinado depois com os 5 folds + --save_models
     (comando sugerido no próprio leaderboard).

Cuidado ao ler o placar: com ~4.000 linhas de validação, o erro-padrão
da acurácia é de ~0,8 ponto. Diferenças menores que ~1,5 ponto entre
duas configurações podem ser só sorte do fold.

Uso:
    python grid_v2.py --data train.xlsx
Edite GRID abaixo antes de rodar (é só uma lista Python).
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# Cada item vira uma rodada. "tag" precisa ser único.
# Tempo aproximado por item na RTX 3060 Ti: ~1h no large, ~20 min no base
# (confira com o --debug; depende do seu ritmo real).
GRID = [
    {"tag": "g_large_lr1e-5_ep3", "args": ["--lr", "1e-5", "--epochs", "3"]},
    {"tag": "g_large_lr2e-5_ep3", "args": ["--lr", "2e-5", "--epochs", "3"]},
    {"tag": "g_large_lr1e-5_ep4", "args": ["--lr", "1e-5", "--epochs", "4"]},
    {"tag": "g_large_lr2e-5_ep2", "args": ["--lr", "2e-5", "--epochs", "2"]},
    {"tag": "g_large_dupone",     "args": ["--lr", "1.5e-5", "--dup_weight", "one"]},
    {"tag": "g_large_len256",     "args": ["--lr", "1.5e-5", "--max_len", "256"]},
    {"tag": "g_legalbert",        "args": ["--model", "felipemaiapolo/legalnlp-bert",
                                           "--lr", "3e-5", "--batch_size", "16", "--grad_accum", "1"]},
    {"tag": "g_base_lr3e-5",      "args": ["--model", "neuralmind/bert-base-portuguese-cased",
                                           "--lr", "3e-5", "--batch_size", "16", "--grad_accum", "1"]},
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="train.xlsx")
    p.add_argument("--kfold", type=int, default=5)
    p.add_argument("--fold", default="0", help="Qual fold usar na avaliação (padrão 0).")
    p.add_argument("--extra", default="",
                   help='Flags repassadas a TODAS as configurações, ex.: '
                        '--extra "--no_grad_ckpt --batch_size 4 --grad_accum 4"')
    a = p.parse_args()

    logs = Path("grid_logs")
    logs.mkdir(exist_ok=True)
    results = []
    t_all = time.time()

    for i, g in enumerate(GRID, 1):
        cmd = [sys.executable, "train_v2.py", "--data", a.data, "--kfold", str(a.kfold),
               "--folds", a.fold, "--tag", g["tag"]] + g["args"] + a.extra.split()
        log = logs / f"{g['tag']}.log"
        print(f"[{i}/{len(GRID)}] {g['tag']} ... (log: {log})", flush=True)
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        mins = (time.time() - t0) / 60

        # train_v2 grava summary.json dentro de runs_v2/<tag>_<timestamp>/
        runs = sorted(Path("runs_v2").glob(f"{g['tag']}_*"), key=lambda d: d.stat().st_mtime)
        summ = runs[-1] / "summary.json" if runs else None
        if rc == 0 and summ and summ.exists():
            m = json.load(open(summ))["overall_oof"]
            results.append((g, m, mins))
            print(f"    acc={m['accuracy']:.4f} macro-F1={m['macro_f1']:.4f} "
                  f"log-loss={m['log_loss']:.4f} ({mins:.0f} min)", flush=True)
        else:
            results.append((g, None, mins))
            print(f"    FALHOU (código {rc}) -- veja {log}. Seguindo para o próximo.", flush=True)

    ok = sorted([r for r in results if r[1]], key=lambda r: (-r[1]["accuracy"], r[1]["log_loss"]))
    lines = [f"=== GRID v2 (fold {a.fold} de {a.kfold}) -- {(time.time() - t_all) / 3600:.1f}h ===",
             "Erro-padrão da acurácia ~0,8 ponto: diferenças < ~1,5 ponto podem ser sorte.",
             "Referência: baseline TF-IDF+LogReg ~0,456\n"]
    for g, m, mins in ok:
        lines.append(f"{g['tag']:24s} acc={m['accuracy']:.4f} macro-F1={m['macro_f1']:.4f} "
                     f"log-loss={m['log_loss']:.4f}  ({mins:.0f} min)")
    for g, m, mins in results:
        if m is None:
            lines.append(f"{g['tag']:24s} FALHOU")
    if ok:
        best = ok[0][0]
        lines.append(f"\nVencedor: {best['tag']}. Para gerar os modelos finais:")
        lines.append(f"python train_v2.py --data {a.data} --kfold {a.kfold} --save_models "
                     f"--tag final {' '.join(best['args'])} {a.extra}".rstrip())
    text = "\n".join(lines)
    Path("leaderboard_v2.txt").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()