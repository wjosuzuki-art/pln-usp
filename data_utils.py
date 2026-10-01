"""
data_utils.py
Funções compartilhadas de carga, limpeza e divisão treino/validação
para o EP1 (classificação de clareza de respostas e-SIC).

Ponto crítico: ~10,6% das respostas do corpus são textos IDÊNTICOS
repetidos (cartas-modelo enviadas a cidadãos diferentes), e o mesmo
texto pode ter rótulos de clareza diferentes (avaliação subjetiva do
próprio usuário). Se você fizer um split aleatório comum, cópias do
mesmo texto podem cair em treino E validação ao mesmo tempo -> o
modelo "decora" em vez de generalizar, e a acurácia de validação fica
artificialmente inflada. Por isso, toda divisão aqui é feita por
GRUPO (agrupando pelo texto exato), nunca por linha.
"""

import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

LABELS = ["c1", "c234", "c5"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}


def clean_text(t: str) -> str:
    """Limpeza mínima: entidades HTML residuais + normalização de espaços.
    Propositalmente leve -- não removemos stopwords/acentos/casing porque
    tokenizers de transformers (e o SetFit) lidam melhor com texto natural."""
    if not isinstance(t, str):
        return ""
    t = re.sub(r"&deg;?", "°", t)
    t = re.sub(r"&[a-zA-Z]+;", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_train(path: str) -> pd.DataFrame:
    """Carrega o train.xlsx (ou .csv), limpa o texto e remove linhas vazias."""
    if path.endswith(".csv"):
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    else:
        df = pd.read_excel(path)

    assert "resp_text" in df.columns and "clarity" in df.columns, (
        f"Esperava colunas resp_text/clarity, achei: {df.columns.tolist()}"
    )

    df["text"] = df["resp_text"].apply(clean_text)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    bad = set(df["clarity"].unique()) - set(LABELS)
    assert not bad, f"Rótulos inesperados encontrados: {bad} (esperado {LABELS})"

    df["label_id"] = df["clarity"].map(LABEL2ID)
    return df


def report_duplicate_leakage_risk(df: pd.DataFrame) -> None:
    """Só um print informativo -- mostra o tamanho do problema de duplicatas."""
    dup_mask = df.duplicated(subset="text", keep=False)
    n_dup_rows = dup_mask.sum()
    n_dup_groups = df.loc[dup_mask, "text"].nunique()
    print(
        f"[data_utils] {n_dup_rows} linhas ({n_dup_rows/len(df):.1%}) pertencem a "
        f"{n_dup_groups} grupos de texto duplicado. Fazendo split por grupo para evitar vazamento."
    )


def group_train_val_split(df: pd.DataFrame, val_size: float = 0.15, seed: int = 42):
    """Split rápido treino/validação (para busca de hiperparâmetros),
    agrupado por texto exato. Use isto no dia a dia; é bem mais barato
    que k-fold completo."""
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    tr_idx, va_idx = next(gss.split(df, groups=df["text"]))
    return df.iloc[tr_idx].reset_index(drop=True), df.iloc[va_idx].reset_index(drop=True)


def group_kfold_splits(df: pd.DataFrame, n_splits: int = 3, seed: int = 42):
    """Gera os folds para a validação cruzada 'oficial' (a que vai pro
    relatório). n_splits=3 por padrão para caber no tempo de treino numa
    GTX 1060 -- suba para 5 se sobrar tempo."""
    gkf = GroupKFold(n_splits=n_splits)
    for tr_idx, va_idx in gkf.split(df, groups=df["text"]):
        yield df.iloc[tr_idx].reset_index(drop=True), df.iloc[va_idx].reset_index(drop=True)


def duplicate_noise_ceiling(df: pd.DataFrame) -> float:
    """Recalcula o teto de acurácia estimado pelo ruído de rótulo
    (ver conversa) -- útil para citar no relatório com o corpus que
    vocês realmente usaram no treino final."""
    dup_mask = df.duplicated(subset="text", keep=False)
    dup = df[dup_mask]
    if len(dup) == 0:
        return 1.0
    grp = dup.groupby("text")["clarity"].agg(lambda s: s.value_counts().max() / len(s))
    sizes = dup.groupby("text").size()
    ceiling_on_dupes = (grp * sizes).sum() / sizes.sum()
    overall = (ceiling_on_dupes * len(dup) + (len(df) - len(dup))) / len(df)
    return overall


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "train_full.xlsx"
    df = load_train(path)
    print(f"Linhas: {len(df)} | distribuição:\n{df['clarity'].value_counts(normalize=True)}")
    report_duplicate_leakage_risk(df)
    print(f"Teto otimista por ruído de rótulo: {duplicate_noise_ceiling(df):.3f}")