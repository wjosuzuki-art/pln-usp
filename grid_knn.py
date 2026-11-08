
import pandas as pd
import numpy as np
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from data_utils import load_train

df = load_train("train_limpo.xlsx")

X = df["text"]
y = df["clarity"]

print(f"Total de exemplos: {len(df)}")
print("\nDistribuição das classes:")
print(y.value_counts())
print("\nTextos únicos:", X.nunique())


n_splits = 3

gkf = GroupKFold(n_splits=n_splits)

folds = list(gkf.split(df, groups=df["text"]))

print(f"\nNúmero de folds: {len(folds)}")

for k, (train_idx, val_idx) in enumerate(folds):
    print(
        f"Fold {k + 1}: "
        f"{len(train_idx)} exemplos de treino | "
        f"{len(val_idx)} exemplos de validação"
    )

    textos_treino = set(X.iloc[train_idx])
    textos_validacao = set(X.iloc[val_idx])

    intersecao = textos_treino.intersection(textos_validacao)

    assert len(intersecao) == 0, (
        f"Vazamento de dados no fold {k + 1}!"
    )

param_grid = [
    # TF-IDF por palavras
    {
        "tfidf__analyzer": ["word"],
        "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
        "tfidf__min_df": [1, 2, 5],
        "tfidf__sublinear_tf": [True, False],

        "knn__n_neighbors": [3, 5, 7, 9, 11, 15],
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["cosine", "euclidean"],
    },

    # TF-IDF por caracteres
    {
        "tfidf__analyzer": ["char"],
        "tfidf__ngram_range": [(3, 9)],
        "tfidf__min_df": [1, 2, 5],
        "tfidf__sublinear_tf": [True, False],

        "knn__n_neighbors": [3, 5, 7, 9, 11, 15],
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["cosine", "euclidean"],
    }
]

combinacoes = list(ParameterGrid(param_grid))

print(f"\nTotal de combinações: {len(combinacoes)}")
print(f"Total de treinamentos: {len(combinacoes) * n_splits}")

resultados = []

inicio_total = time.time()

arquivo_txt = "resultados_knn.txt"

with open(arquivo_txt, "w", encoding="utf-8") as f:

    f.write("GRID SEARCH KNN - GROUP K-FOLD\n")
    f.write("=" * 70 + "\n")
    f.write(f"Total de exemplos: {len(df)}\n")
    f.write(f"Número de folds: {n_splits}\n")
    f.write(f"Combinações: {len(combinacoes)}\n")
    f.write(f"Treinamentos: {len(combinacoes) * n_splits}\n")
    f.write("=" * 70 + "\n\n")

    for i, params in enumerate(combinacoes):

        inicio_config = time.time()

        config = str(params)

        acuracias = []

        for fold, (train_idx, val_idx) in enumerate(folds):

            inicio_fold = time.time()

            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]

            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            pipeline = Pipeline([
                ("tfidf", TfidfVectorizer()),
                ("knn", KNeighborsClassifier())
            ])

            pipeline.set_params(**params)

            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_val)

            acc = accuracy_score(y_val, y_pred)

            acuracias.append(acc)

            tempo_fold = time.time() - inicio_fold

            resultados.append({
                "configuracao": config,
                "combinacao": i + 1,
                "fold": fold + 1,
                "accuracy": acc,
                "tempo_segundos": tempo_fold
            })

            mensagem = (
                f"Combinação {i + 1}/{len(combinacoes)} | "
                f"Fold {fold + 1}/{n_splits} | "
                f"Accuracy: {acc:.5f} | "
                f"Tempo: {tempo_fold:.2f}s\n"
                f"Parâmetros: {params}\n"
                + "-" * 70 + "\n"
            )

            print(mensagem)
            f.write(mensagem)
            f.flush()

        media = np.mean(acuracias)
        desvio = np.std(acuracias)

        tempo_config = time.time() - inicio_config

        mensagem = (
            f"\nRESULTADO DA COMBINAÇÃO {i + 1}\n"
            f"Accuracy média: {media:.5f}\n"
            f"Desvio-padrão: {desvio:.5f}\n"
            f"Tempo total: {tempo_config:.2f}s\n"
            f"Parâmetros: {params}\n"
            + "=" * 70 + "\n\n"
        )

        print(mensagem)
        f.write(mensagem)
        f.flush()


resultados_df = pd.DataFrame(resultados)

resultados_df.to_csv(
    "resultados_knn_folds.csv",
    index=False,
    encoding="utf-8-sig"
)

resumo = (
    resultados_df
    .groupby(["combinacao", "configuracao"])
    .agg(
        accuracy_media=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        tempo_medio=("tempo_segundos", "mean")
    )
    .reset_index()
)

resumo = resumo.sort_values(
    by="accuracy_media",
    ascending=False
)

resumo.to_csv(
    "resultados_knn_resumo.csv",
    index=False,
    encoding="utf-8-sig"
)

melhor = resumo.iloc[0]

tempo_total = time.time() - inicio_total

print("\n" + "=" * 70)
print("MELHOR CONFIGURAÇÃO")
print("=" * 70)

print(f"Accuracy média: {melhor['accuracy_media']:.5f}")
print(f"Desvio-padrão: {melhor['accuracy_std']:.5f}")
print(f"Tempo médio por fold: {melhor['tempo_medio']:.2f}s")

print("\nParâmetros:")
print(melhor["configuracao"])

print(f"\nTempo total: {tempo_total / 60:.2f} minutos")

print("\nArquivos salvos:")
print("- resultados_knn.txt")
print("- resultados_knn_folds.csv")
print("- resultados_knn_resumo.csv")