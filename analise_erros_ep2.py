
import pandas as pd

arquivo_gabarito = "teste1 - gabarito.xlsx"
arquivo_nosso = "teste 1 - nosso_ep2.xlsx"

gabarito = pd.read_excel(arquivo_gabarito)
nosso = pd.read_excel(arquivo_nosso)

def manter_textos_sem_conflito(df):

    contagem = df.groupby("resp_text")["clarity"].nunique()

    textos_validos = contagem[contagem == 1].index

    return df[df["resp_text"].isin(textos_validos)].drop_duplicates(
        subset=["resp_text"]
    )


gabarito_limpo = manter_textos_sem_conflito(gabarito)
nosso_limpo = manter_textos_sem_conflito(nosso)

comparacao = pd.merge(
    nosso_limpo,
    gabarito_limpo,
    on="resp_text",
    how="inner",
    suffixes=("_nosso", "_gabarito"),
    validate="one_to_one"
)

combinacoes = [
    ("c1", "c5"),
    ("c5", "c1"),
    ("c1", "c234"),
    ("c234", "c1"),
    ("c5", "c234"),
    ("c234", "c5")
]

resultados = []

print("\n" + "=" * 100)
print("ANÁLISE DE DIVERGÊNCIAS")
print("=" * 100)

for i, (cn, cg) in enumerate(combinacoes, 1):

    casos = comparacao[
        (comparacao["clarity_nosso"] == cn) &
        (comparacao["clarity_gabarito"] == cg)
    ]

    print(f"\nCASO {i}: NOSSO = {cn} | GABARITO = {cg}")

    if casos.empty:
        print("Nenhum caso encontrado.")
        continue

    caso = casos.iloc[0]

    print("\nRESP_TEXT:")
    print(caso["resp_text"])

    print("\nCLARITY NO NOSSO ARQUIVO:", caso["clarity_nosso"])
    print("CLARITY NO GABARITO:", caso["clarity_gabarito"])

    resultados.append({
        "resp_text": caso["resp_text"],
        "clarity_nosso": caso["clarity_nosso"],
        "clarity_gabarito": caso["clarity_gabarito"]
    })

    print("-" * 100)

print("\nRESUMO FINAL")
print("Registros originais no gabarito:", len(gabarito))
print("Registros originais no nosso arquivo:", len(nosso))
print("Textos válidos no gabarito:", len(gabarito_limpo))
print("Textos válidos no nosso arquivo:", len(nosso_limpo))
print("Textos comparados:", len(comparacao))
print("Divergências encontradas:", len(resultados), "de 6")

if resultados:
    pd.DataFrame(resultados).to_excel(
        "divergencias_encontradas.xlsx",
        index=False
    )
    print("\nArquivo salvo: divergencias_encontradas.xlsx")

print("\nAnálise concluída!")
