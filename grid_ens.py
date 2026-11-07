"""
grid_ens.py -- aproveita as execuções que os grids JÁ deixaram prontas (sem usar a GPU).

Cada configuração concluída do grid3f.py tem previsões fora da amostra (OOF) nos folds 0-2 e
previsões do teste. Aqui medimos, nessas mesmas linhas, se MÉDIA de várias execuções do mesmo modelo
(sementes e configurações) supera uma execução sozinha -- e a média entre modelos supera a base.

Regra fixada ANTES de olhar os resultados (para não escolher o subconjunto que "deu certo"):
  * 'sementes' = a base e suas repetições (_ref, _seed2, _seed3);
  * 'filtrada' = todas as configurações com log-loss <= --max_ll (corta as muito confiantes,
    como o NorBERTo sem LLRD, que tinha log-loss 1,21);
  * 'todas' = sem filtro.

Uso:
    python grid_ens.py --data train_mais_test1.xlsx
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data_utils import LABEL2ID, load_train          # noqa: E402
from grid3f import FAMILIAS, paired_bootstrap        # noqa: E402

COLS = ["p_c1", "p_c234", "p_c5"]


def runs_da_familia(prefix, n_folds=3):
    """tag -> pasta, só execuções concluídas com 3 folds (exclui *_f34 e *_5f_*)."""
    pat = re.compile(rf"^{re.escape(prefix)}_[^_]+_\d+$")
    best = {}
    for d in Path("runs_v2").glob(f"{prefix}_*"):
        if pat.match(d.name) and (d / "summary.json").exists():
            if len(json.load(open(d / "summary.json")).get("folds", [])) == n_folds:
                tag = d.name.rsplit("_", 1)[0]
                if tag not in best or d.stat().st_mtime > best[tag].stat().st_mtime:
                    best[tag] = d
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="train_mais_test1.xlsx")
    p.add_argument("--familias", nargs="*", default=["legal", "norb", "large"])
    p.add_argument("--folds", default="0,1,2")
    p.add_argument("--max_ll", type=float, default=1.10, help="log-loss máximo da 'filtrada'")
    a = p.parse_args()

    df = load_train(a.data)
    y = df["clarity"].map(LABEL2ID).values
    texts = df["text"].values
    sel = [int(x) for x in a.folds.split(",")]

    fold_ref, fam = None, {}
    for f in a.familias:
        P = FAMILIAS[f]["prefix"]
        probs = {}
        for tag, d in runs_da_familia(P, len(sel)).items():
            o = pd.read_csv(d / "oof_probs.csv")
            if len(o) != len(df):
                continue
            if fold_ref is None:
                fold_ref = o["fold"].values
            elif not np.array_equal(fold_ref, o["fold"].values):
                continue
            probs[tag] = (o[COLS].values, d)
        if probs:
            fam[f] = probs
    assert fam, "nenhuma execução de grid encontrada em runs_v2/ (rode na pasta do projeto)"
    mask = np.isin(fold_ref, sel)
    ym, tm = y[mask], texts[mask]

    acc = lambda Pm: float((Pm.argmax(1) == ym).mean())
    ll = lambda Pm: float(-np.log(np.clip(Pm[np.arange(len(ym)), ym], 1e-7, 1)).mean())
    correct = lambda Pm: Pm.argmax(1) == ym
    print(f"Linhas avaliadas: {mask.sum()} (folds {a.folds}) | regra 'filtrada': log-loss <= {a.max_ll}\n")

    combos, test_out = {}, {}
    for f, probs in fam.items():
        name = FAMILIAS[f]["nome"]
        P = FAMILIAS[f]["prefix"]
        M = {t: v[0][mask] for t, v in probs.items()}
        ref = M.get(f"{P}_ref")
        print(f"=== {name}: {len(M)} execuções ===")
        for t in sorted(M, key=lambda t: -acc(M[t])):
            print(f"  {t:14s} acc={acc(M[t]) * 100:6.2f}%  log-loss={ll(M[t]):.4f}")
        groups = {
            "sementes": [t for t in M if re.search(r"_(ref|seed\d)$", t)],
            "filtrada": [t for t in M if ll(M[t]) <= a.max_ll],
            "todas": list(M),
        }
        for gname, tags in groups.items():
            if len(tags) < 2:
                continue
            avg = np.mean([M[t] for t in tags], axis=0)
            combos[(f, gname)] = avg
            line = f"  MÉDIA {gname:9s} ({len(tags):2d} exec.) acc={acc(avg) * 100:6.2f}%  log-loss={ll(avg):.4f}"
            if ref is not None:
                d = acc(avg) - acc(ref)
                lo, hi = paired_bootstrap(correct(avg), correct(ref), tm)
                line += f"   vs {P}_ref: {d * 100:+.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}]{' *' if lo > 0 else ''}"
            print(line)
            # previsões do teste desta média
            tps = []
            for t in tags:
                for k in sel:
                    fp = probs[t][1] / f"test_probs_fold{k}.npy"
                    if fp.exists():
                        tps.append(np.load(fp))
            if tps:
                test_out[(f, gname)] = np.mean(tps, axis=0)
        for gname in groups:                      # família com só 1 execução: ela própria entra na comparação entre modelos
            if (f, gname) not in combos and ref is not None:
                combos[(f, gname)] = ref
        print()

    refs = {f: probs[f"{FAMILIAS[f]['prefix']}_ref"][0][mask] for f, probs in fam.items()
            if f"{FAMILIAS[f]['prefix']}_ref" in probs}
    if len(refs) >= 2:
        base = np.mean(list(refs.values()), axis=0)
        print("=== ENTRE MODELOS (sem TF-IDF; média simples das probabilidades) ===")
        print(f"  base: média das {len(refs)} receitas '_ref'       acc={acc(base) * 100:6.2f}%  log-loss={ll(base):.4f}")
        for gname in ("sementes", "filtrada", "todas"):
            mem = [combos[(f, gname)] for f in fam if (f, gname) in combos]
            if len(mem) < len(fam):
                continue
            avg = np.mean(mem, axis=0)
            d = acc(avg) - acc(base)
            lo, hi = paired_bootstrap(correct(avg), correct(base), tm)
            print(f"  média das famílias ({gname:9s})              acc={acc(avg) * 100:6.2f}%  log-loss={ll(avg):.4f}"
                  f"   vs base: {d * 100:+.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}]{' *' if lo > 0 else ''}")
        print("\n  * = IC 95% pareado (por texto) sem zero. Se o ganho entre modelos for < ~0,3 ponto, as horas de GPU")
        print("    rendem mais em modelos NOVOS (arquiteturas diferentes) do que em mais sementes/configurações.")

    for (f, g), tp in test_out.items():
        np.save(f"grid_ens_test_{f}_{g}.npy", tp)
    if test_out:
        print(f"\nPrevisões do teste das médias salvas em grid_ens_test_<familia>_<regra>.npy "
              f"({len(test_out)} arquivos).")


if __name__ == "__main__":
    main()
