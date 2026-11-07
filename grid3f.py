"""
grid3f.py -- grid de 3 folds para UM modelo por vez (BERTimbau-large, LegalNLP-BERT ou NorBERTo-large),
pensado para o treino ampliado (train_mais_test1.xlsx) e para montar o ensemble do teste novo.

Diferenças em relação ao grid_large3f.py / grid_norb3f.py:
  * A BASE de cada modelo é a MELHOR receita que já medimos para ele (não a do outro grupo):
      - BERTimbau-large: vencedora do grid anterior (L_ls05: lr 1e-5, 4 ép., 256 tokens, lote 32,
        AdamW padrão, sem LLRD, label smoothing 0,05, rótulo duro);
      - NorBERTo-large: nossa receita final (lr 3e-5, 3 ép., 512 início+fim, LLRD 0,9, LS 0,05, alvo suave)
        -- a receita do outro grupo fez o NorBERTo decorar o treino (log-loss ~1,9);
      - LegalNLP-BERT: nossa receita final (lr 3e-5, lote 16, 3 ép., 512 início+fim, LLRD, LS, alvo suave).
  * Cada configuração muda UMA coisa; folds 0,1,2 da divisão em 5; uma 2ª semente mede o ruído.
  * COMBO das mudanças que ajudaram mais que o ruído (se 2+). Em conflito (ex.: duas lr), vale a melhor.
  * CONFIRMAÇÃO nos folds 3,4 (não usados na escolha) do vencedor E da base -> comparação sem viés;
    os dois viram pastas de 5 folds (<tag>_5f_*) prontas para o ensemble.py.
  * Retomável e com orçamento de horas (pula o que não cabe, guardando tempo para a confirmação).

Uso:
    python grid3f.py --familia legal --data train_mais_test1.xlsx --test test2.xlsx --budget_h 10
    python grid3f.py --familia norb  --data train_mais_test1.xlsx --test test2.xlsx --budget_h 24
    python grid3f.py --familia large --data train_mais_test1.xlsx --test test2.xlsx --budget_h 14
    python grid3f.py --familia norb  --data train_mais_test1.xlsx --report_only
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

GRID_FOLDS, CONFIRM_FOLDS, KFOLD = "0,1,2", "3,4", 5
HELP_THRESHOLD = 0.003
LEN512_FACTOR = 1.3

_COMMON = {"--warmup": "0.1", "--weight_decay": "0.01", "--seed": "42"}
_OURS = {"--max_len": "512", "--head_tokens": "128", "--llrd": "0.9", "--label_smoothing": "0.05",
         "--dup_weight": "sqrt", "--adam_beta2": "0.98", "--adam_eps": "1e-6"}

FAMILIAS = {
    "large": {
        "nome": "BERTimbau-large", "prefix": "B2", "rate": 9.7,
        "base": {**_COMMON, "--model": "neuralmind/bert-large-portuguese-cased", "--lr": "1e-5", "--epochs": "4",
                 "--max_len": "256", "--head_tokens": "512", "--batch_size": "8", "--grad_accum": "4",
                 "--llrd": "1.0", "--label_smoothing": "0.05", "--dup_weight": "count",
                 "--adam_beta2": "0.999", "--adam_eps": "1e-8"},
        "configs": [
            ("ref",    {},                                           "vencedora do grid anterior (L_ls05)"),
            ("seed2",  {"--seed": "1042"},                           "outra semente (mede o ruído)"),
            ("ls10",   {"--label_smoothing": "0.1"},                 "label smoothing 0,10"),
            ("soft",   {"--dup_weight": "sqrt"},                     "duplicatas: alvo suave, peso raiz(n)"),
            ("ep3",    {"--epochs": "3"},                            "3 épocas"),
            ("lr15",   {"--lr": "1.5e-5"},                           "lr 1,5e-5"),
            ("len512", {"--max_len": "512", "--head_tokens": "128"}, "512 tokens, início+fim"),
            ("ep5",    {"--epochs": "5"},                            "5 épocas"),
        ],
    },
    "norb": {
        "nome": "NorBERTo-large", "prefix": "N2", "rate": 11.2,
        "base": {**_COMMON, **_OURS, "--model": "Itau-Unibanco/NorBERTo-large", "--lr": "3e-5", "--epochs": "3",
                 "--batch_size": "8", "--grad_accum": "2"},
        "configs": [
            ("ref",     {},                                          "nossa receita final do NorBERTo"),
            ("seed2",   {"--seed": "1042"},                          "outra semente (mede o ruído)"),
            ("ls10",    {"--label_smoothing": "0.1"},                "label smoothing 0,10"),
            ("ep2",     {"--epochs": "2"},                           "2 épocas"),
            ("lr4e-5",  {"--lr": "4e-5"},                            "lr 4e-5"),
            ("len256",  {"--max_len": "256", "--head_tokens": "512"}, "256 tokens, só o início"),
            ("lr2e-5",  {"--lr": "2e-5"},                            "lr 2e-5"),
            ("hard",    {"--dup_weight": "count"},                   "rótulo duro (repete as cópias)"),
            ("nollrd",  {"--llrd": "1.0"},                           "sem LLRD"),
            ("bs32",    {"--grad_accum": "4"},                       "lote efetivo 32"),
            ("beta999", {"--adam_beta2": "0.999", "--adam_eps": "1e-8"}, "AdamW padrão"),
            ("ep4",     {"--epochs": "4"},                           "4 épocas"),
        ],
    },
    "legal": {
        "nome": "LegalNLP-BERT", "prefix": "J2", "rate": 2.9,
        "base": {**_COMMON, **_OURS, "--model": "felipemaiapolo/legalnlp-bert", "--lr": "3e-5", "--epochs": "3",
                 "--batch_size": "16", "--grad_accum": "1"},
        "configs": [
            ("ref",     {},                                          "nossa receita final do LegalNLP"),
            ("seed2",   {"--seed": "1042"},                          "outra semente (mede o ruído)"),
            ("lr2e-5",  {"--lr": "2e-5"},                            "lr 2e-5"),
            ("lr5e-5",  {"--lr": "5e-5"},                            "lr 5e-5"),
            ("ep2",     {"--epochs": "2"},                           "2 épocas"),
            ("ep4",     {"--epochs": "4"},                           "4 épocas"),
            ("ls10",    {"--label_smoothing": "0.1"},                "label smoothing 0,10"),
            ("ls0",     {"--label_smoothing": "0"},                  "sem label smoothing"),
            ("len256",  {"--max_len": "256", "--head_tokens": "512"}, "256 tokens, só o início"),
            ("hard",    {"--dup_weight": "count"},                   "rótulo duro (repete as cópias)"),
            ("nollrd",  {"--llrd": "1.0"},                           "sem LLRD"),
            ("bs32",    {"--grad_accum": "2"},                       "lote efetivo 32"),
            ("beta999", {"--adam_beta2": "0.999", "--adam_eps": "1e-8"}, "AdamW padrão"),
            ("seed3",   {"--seed": "2042"},                          "3ª semente (ruído)"),
        ],
    },
}


# ───────────────────────────── utilidades ─────────────────────────────

def cfg_args(base, over):
    d = dict(base)
    d.update(over)
    out = []
    for k, v in d.items():
        out += [k, v]
    return out, d


def completed_run(tag, n_folds_expected):
    pat = re.compile(rf"^{re.escape(tag)}_\d+$")
    for d in sorted(Path("runs_v2").glob(f"{tag}_*"), key=lambda d: -d.stat().st_mtime):
        if pat.match(d.name) and (d / "summary.json").exists():
            s = json.load(open(d / "summary.json"))
            if len(s.get("folds", [])) == n_folds_expected:
                return d
    return None


def size_factor(d):
    return LEN512_FACTOR if int(d["--max_len"]) > 256 else 1.0


def estimate_min(d, n_folds, rate):
    return rate * n_folds * int(d["--epochs"]) * size_factor(d)


def measured_rate(done, default):
    vals = []
    for d, run_dir in done:
        s = json.load(open(run_dir / "summary.json"))
        nf = len(s["folds"])
        if nf and s.get("minutes"):
            vals.append(s["minutes"] / (nf * int(d["--epochs"]) * size_factor(d)))
    return float(np.median(vals)) if vals else default


def run_train(tag, d_args, folds, data, test, log):
    cmd = [sys.executable, "train_v2.py", "--data", data, "--kfold", str(KFOLD),
           "--folds", folds, "--tag", tag] + d_args
    if test:
        cmd += ["--test", test]
    with open(log, "w", encoding="utf-8") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


# ───────────────────────────── análise ─────────────────────────────

def paired_bootstrap(correct_a, correct_b, clusters, n_boot=2000, seed=0):
    _, cid = np.unique(clusters, return_inverse=True)
    nc = cid.max() + 1
    diff_c = np.bincount(cid, weights=correct_a.astype(float) - correct_b.astype(float), minlength=nc)
    cnt_c = np.bincount(cid, minlength=nc).astype(float)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        w = np.bincount(rng.integers(0, nc, nc), minlength=nc)
        stats[i] = (w * diff_c).sum() / (w * cnt_c).sum()
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def analyze(entries, df, base_tag, seed_tag=None, folds=(0, 1, 2)):
    from data_utils import LABEL2ID
    y = df["clarity"].map(LABEL2ID).values
    texts = df["text"].values
    res, fold_ref = {}, None
    for tag, desc, rd in entries:
        o = pd.read_csv(Path(rd) / "oof_probs.csv")
        fold, P = o["fold"].values, o[["p_c1", "p_c234", "p_c5"]].values
        if len(fold) != len(df):
            print(f"[aviso] {tag}: {len(fold)} linhas no OOF, esperado {len(df)} (outro arquivo de treino?) -- ignorado")
            continue
        if fold_ref is None:
            fold_ref = fold
        elif not np.array_equal(fold, fold_ref):
            print(f"[aviso] {tag}: divisão de folds diferente -- ignorado")
            continue
        mask = np.isin(fold, folds) & ~np.isnan(P).any(1)
        if not np.isin(folds, np.unique(fold[mask])).all():
            continue
        correct = P.argmax(1) == y
        ll = -np.log(np.clip(P[mask][np.arange(mask.sum()), y[mask]], 1e-7, 1)).mean()
        res[tag] = {"desc": desc, "correct": correct, "per_fold": [correct[fold == k].mean() for k in folds],
                    "acc_rows": float(correct[mask].mean()), "log_loss": float(ll)}
    if not res:
        return ["(nenhuma execução concluída ainda)"], res
    lines = []
    if base_tag not in res and seed_tag in res:
        lines.append(f"[aviso] {base_tag} não concluída -- usando {seed_tag} como base da comparação.\n")
        base_tag = seed_tag
    allmask = np.isin(fold_ref, folds)
    lines.append(f"{'config':14s} " + " ".join(f"fold{k:<4d}" for k in folds) +
                 f"   média   log-loss   vs {base_tag:10s} [IC 95%]     descrição")
    for t in sorted(res, key=lambda t: -res[t]["acc_rows"]):
        r, cmp = res[t], ""
        if base_tag in res and t != base_tag:
            d = r["acc_rows"] - res[base_tag]["acc_rows"]
            lo, hi = paired_bootstrap(r["correct"][allmask], res[base_tag]["correct"][allmask], texts[allmask])
            cmp = f"{d * 100:+.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}]{' *' if lo > 0 or hi < 0 else ''}"
            r["delta"] = d
        lines.append(f"{t:14s} " + " ".join(f"{a * 100:6.2f}%" for a in r["per_fold"]) +
                     f"  {r['acc_rows'] * 100:6.2f}%  {r['log_loss']:.4f}   {cmp:28s} {r['desc']}")
    lines.append("\n* = IC 95% sem zero (sem correção para comparações múltiplas: trate como indício).")
    seeds = [res[t]["acc_rows"] for t in res if t.split("_", 1)[-1] in ("ref", "seed2", "seed3")]
    if len(seeds) >= 2:
        lines.append(f"Ruído de semente (maior - menor entre as sementes da base): {(max(seeds) - min(seeds)) * 100:.2f} "
                     f"ponto. Diferenças desse tamanho são indistinguíveis de sorte.")
    lines.append(f"Linhas avaliadas: {int(allmask.sum())} (folds {','.join(map(str, folds))} da divisão em 5).")
    return lines, res


def merge_5f(tag, rd_a, rd_b):
    out = Path("runs_v2") / f"{tag}_5f_{int(time.time())}"
    out.mkdir(parents=True)
    oa = pd.read_csv(Path(rd_a) / "oof_probs.csv")
    ob = pd.read_csv(Path(rd_b) / "oof_probs.csv")
    cols = ["p_c1", "p_c234", "p_c5"]
    fill = oa[cols].isna().any(axis=1)
    oa.loc[fill, cols] = ob.loc[fill, cols].values
    oa.to_csv(out / "oof_probs.csv", index=False)
    tps = sorted(list(Path(rd_a).glob("test_probs_fold*.npy")) + list(Path(rd_b).glob("test_probs_fold*.npy")))
    if len(tps) == KFOLD:
        np.save(out / "test_probs_mean.npy", np.mean([np.load(p) for p in tps], axis=0))
    cfg = json.load(open(Path(rd_a) / "config.json"))
    cfg["folds"], cfg["merged_from"] = None, [str(rd_a), str(rd_b)]
    json.dump(cfg, open(out / "config.json", "w"), indent=2)
    sa, sb = json.load(open(Path(rd_a) / "summary.json")), json.load(open(Path(rd_b) / "summary.json"))
    json.dump({"folds": sa["folds"] + sb["folds"], "minutes": sa["minutes"] + sb["minutes"],
               "rows_covered": int((~oa[cols].isna().any(axis=1)).sum())},
              open(out / "summary.json", "w"), indent=2)
    return out, len(tps) == KFOLD


# ───────────────────────────── principal ─────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--familia", required=True, choices=list(FAMILIAS))
    p.add_argument("--data", default="train_mais_test1.xlsx")
    p.add_argument("--test", default=None, help="arquivo de teste: salva previsões de cada fold")
    p.add_argument("--budget_h", type=float, default=24.0)
    p.add_argument("--no_combo", action="store_true")
    p.add_argument("--no_confirm", action="store_true")
    p.add_argument("--report_only", action="store_true")
    a = p.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from data_utils import load_train
    df = load_train(a.data)

    F = FAMILIAS[a.familia]
    P, base = F["prefix"], F["base"]
    configs = [(f"{P}_{t}", o, desc) for t, o, desc in F["configs"]]
    base_tag, seed_tag, combo_tag = f"{P}_ref", f"{P}_seed2", f"{P}_combo"
    over_of = {t: o for t, o, _ in configs}

    Path("grid_logs").mkdir(exist_ok=True)
    t_all = time.time()
    elapsed_h = lambda: (time.time() - t_all) / 3600
    nf = len(GRID_FOLDS.split(","))
    done = []
    reserve = lambda rate: 0 if a.no_confirm else 2 * estimate_min(base, 2, rate) / 60

    # ---------- 1. grid ----------
    for i, (tag, over, desc) in enumerate(configs, 1):
        args_list, d = cfg_args(base, over)
        rd = completed_run(tag, nf)
        if rd:
            print(f"[{i}/{len(configs)}] {tag}: já concluída ({rd}) -- pulando", flush=True)
            done.append((d, rd))
            continue
        if a.report_only:
            continue
        rate = measured_rate(done, F["rate"])
        est_h = estimate_min(d, nf, rate) / 60
        if elapsed_h() + est_h + reserve(rate) > a.budget_h:
            print(f"[{i}/{len(configs)}] {tag}: PULADA (estimativa {est_h:.1f}h não cabe: {elapsed_h():.1f}h "
                  f"usadas de {a.budget_h}h, reservando a confirmação)", flush=True)
            continue
        log = Path("grid_logs") / f"{tag}.log"
        print(f"[{i}/{len(configs)}] {tag} ({desc}) -- estimativa {est_h:.1f}h | log: {log}", flush=True)
        t0 = time.time()
        rc = run_train(tag, args_list, GRID_FOLDS, a.data, a.test, log)
        rd = completed_run(tag, nf)
        if rc == 0 and rd:
            s = json.load(open(rd / "summary.json"))
            print(f"    acc={s['overall_oof']['accuracy']:.4f} log-loss={s['overall_oof']['log_loss']:.4f} "
                  f"({(time.time() - t0) / 60:.0f} min)", flush=True)
            done.append((d, rd))
        else:
            print(f"    FALHOU (código {rc}) -- veja {log}. Seguindo.", flush=True)

    def entries_now():
        e = [(t, desc, completed_run(t, nf)) for t, _, desc in configs]
        e.append((combo_tag, "combinação das mudanças que ajudaram", completed_run(combo_tag, nf)))
        return [x for x in e if x[2]]

    # ---------- 2. combo ----------
    lines, res = analyze(entries_now(), df, base_tag, seed_tag)
    seeds = [res[t]["acc_rows"] for t in res if t.split("_", 1)[-1] in ("ref", "seed2", "seed3")]
    thr = max(HELP_THRESHOLD, (max(seeds) - min(seeds)) if len(seeds) >= 2 else 0)
    helped = sorted([(res[t]["delta"], t) for t, _, _ in configs
                     if t in res and "delta" in res[t] and not re.search(r"_(ref|seed\d)$", t)
                     and res[t]["delta"] >= thr])
    combo_over = {}
    for _, t in helped:                      # em ordem crescente de ganho: em conflito, vale a melhor
        combo_over.update(over_of[t])
    cfgfile = Path("grid_logs") / f"{combo_tag}_config.json"
    if cfgfile.exists():
        combo_over = json.load(open(cfgfile))["over"]
    if not a.report_only and not a.no_combo and len(helped) >= 2 and not completed_run(combo_tag, nf):
        args_list, d = cfg_args(base, combo_over)
        rate = measured_rate(done, F["rate"])
        est_h = estimate_min(d, nf, rate) / 60
        if elapsed_h() + est_h + reserve(rate) <= a.budget_h:
            print(f"[combo] base + {[t for _, t in helped]} -- estimativa {est_h:.1f}h", flush=True)
            json.dump({"over": combo_over, "from": [t for _, t in helped]}, open(cfgfile, "w"), indent=2)
            run_train(combo_tag, args_list, GRID_FOLDS, a.data, a.test, Path("grid_logs") / f"{combo_tag}.log")
        else:
            print(f"[combo] pulado: não cabe no orçamento ({est_h:.1f}h)", flush=True)
    elif len(helped) < 2:
        print(f"[combo] menos de 2 mudanças ajudaram (>= +{thr * 100:.2f} ponto) -- nada a combinar.", flush=True)
    over_of[combo_tag] = combo_over
    lines, res = analyze(entries_now(), df, base_tag, seed_tag)

    # ---------- 3. confirmação: vencedor e base nos folds 3,4 ----------
    conf_lines, merged_dirs = [], {}
    cand = sorted(res, key=lambda t: -res[t]["acc_rows"])
    if cand:
        win = cand[0]
        if re.search(r"_seed\d$", win):      # semente "vencedora" = mesma receita da base
            win = base_tag
        to_confirm = [win] + ([base_tag] if win != base_tag and base_tag in res else [])
        conf_entries = []
        for t in to_confirm:
            args_list, d = cfg_args(base, over_of[t])
            ctag = f"{t}_f34"
            rd_c = completed_run(ctag, 2)
            if not rd_c and not a.report_only and not a.no_confirm:
                rate = measured_rate(done, F["rate"])
                est_h = estimate_min(d, 2, rate) / 60
                if elapsed_h() + est_h <= a.budget_h + 0.5:
                    print(f"[confirmação] {t} nos folds {CONFIRM_FOLDS} -- estimativa {est_h:.1f}h", flush=True)
                    run_train(ctag, args_list, CONFIRM_FOLDS, a.data, a.test, Path("grid_logs") / f"{ctag}.log")
                    rd_c = completed_run(ctag, 2)
                else:
                    print(f"[confirmação] {t}: pulada, não cabe ({est_h:.1f}h)", flush=True)
            if rd_c:
                conf_entries.append((t, "vencedor do grid" if t == win else "base", rd_c))
                merged = sorted(Path("runs_v2").glob(f"{t}_5f_*"))
                if not merged:
                    out, _ = merge_5f(t, completed_run(t, nf), rd_c)
                    merged = [out]
                merged_dirs[t] = merged[-1]
        if conf_entries:
            sub, _ = analyze(conf_entries, df, base_tag, folds=(3, 4))
            conf_lines = ["\n=== CONFIRMAÇÃO: folds 3 e 4 (não usados na escolha -> estimativa sem viés) ==="] + sub
            conf_lines.append("\nPastas de 5 folds prontas para o ensemble.py:")
            for t, dpath in merged_dirs.items():
                has_test = (dpath / "test_probs_mean.npy").exists()
                conf_lines.append(f"  {t:14s} {dpath}{'' if has_test else '   (sem previsões de teste: faltou --test)'}")
            conf_lines.append("Use o vencedor; se na confirmação ele não superar a base, prefira a base.")

    text = "\n".join([f"=== GRID {F['nome']}, 3 folds (0,1,2 de 5) | dados: {a.data} ({len(df)} linhas) | "
                      f"{elapsed_h():.1f}h nesta execução ===\n"] + lines + conf_lines)
    Path(f"leaderboard_{a.familia}_3f.txt").write_text(text, encoding="utf-8")
    print("\n" + text, flush=True)


if __name__ == "__main__":
    main()
