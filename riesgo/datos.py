"""Del archivo original de Kaggle a las tablas que usan los modelos.

Hace tres cosas:
  1. construye la variable objetivo "malo" a partir de loan_status (tabla del enunciado);
  2. deja solo lo que se sabe en el momento de pedir el credito, mas unas pocas
     variables derivadas faciles de explicar;
  3. calcula la LGD de los creditos castigados (Charged Off).
"""

from pathlib import Path

import pandas as pd

# Codificacion de loan_status segun el enunciado. Lo que no esta aqui (Current,
# Issued, In Grace Period, Late 16-30) queda como NA y no se usa.
MALO = {
    "Fully Paid": 0,
    "Does not meet the credit policy. Status:Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1,
    "Late (31-120 days)": 1,
    "Does not meet the credit policy. Status:Charged Off": 1,
}

# Estados en los que el cobro ya termino y las recuperaciones son finales.
# Default y Late (31-120 days) son malos, pero todavia no tienen recuperaciones
# (recoveries = 0 en el 100 % de los casos), asi que su LGD aun no se conoce.
CASTIGADOS = ["Charged Off", "Does not meet the credit policy. Status:Charged Off"]

# Columnas del archivo original que se leen. Las de pagos (total_rec_prncp,
# recoveries, ...) solo se usan para construir la LGD, nunca como entradas.
COLUMNAS = [
    "id", "loan_status", "issue_d",
    # de la solicitud y del reporte de credito
    "loan_amnt", "term", "emp_length", "home_ownership", "annual_inc",
    "verification_status", "purpose", "addr_state", "dti", "delinq_2yrs",
    "earliest_cr_line", "inq_last_6mths", "mths_since_last_delinq",
    "mths_since_last_record", "open_acc", "pub_rec", "revol_bal", "revol_util",
    "total_acc", "initial_list_status", "collections_12_mths_ex_med", "acc_now_delinq",
    # asignadas por Lending Club (no se usan por defecto, ver config.yaml)
    "grade", "sub_grade", "int_rate", "installment",
    # para la LGD y la EAD
    "funded_amnt", "total_rec_prncp", "recoveries", "collection_recovery_fee",
]


def leer_original(ruta):
    return pd.read_csv(ruta, usecols=COLUMNAS, low_memory=False)


def crear_objetivo(df):
    """Agrega 'malo' (0/1) y quita los creditos sin desenlace conocido."""
    df = df.copy()
    df["malo"] = df["loan_status"].map(MALO)
    return df.dropna(subset=["malo"]).astype({"malo": int})


def variables_derivadas(df):
    """Variables nuevas, cada una con una razon sencilla."""
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
    df["anio"] = df["fecha"].dt.year

    # "36 months" -> 36
    df["plazo_meses"] = df["term"].str.extract(r"(\d+)").astype(float)

    # "< 1 year" -> 0, "10+ years" -> 10, "n/a" -> NaN
    texto = df["emp_length"].str.replace("< 1", "0", regex=False)
    df["antiguedad_empleo"] = texto.str.extract(r"(\d+)").astype(float)

    # meses entre la primera linea de credito y la fecha del prestamo:
    # quien tiene mas historia crediticia deja mas informacion sobre si mismo
    primera = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y")
    df["antiguedad_credito_meses"] = ((df["fecha"] - primera).dt.days / 30.44).round()

    # En estas dos, NaN no es un dato perdido: significa "nunca ha pasado".
    # Se guarda esa informacion en una columna aparte antes de imputar.
    df["meses_desde_mora"] = df["mths_since_last_delinq"]
    df["meses_desde_registro"] = df["mths_since_last_record"]
    df["nunca_en_mora"] = df["mths_since_last_delinq"].isna().astype(int)
    df["sin_registros_publicos"] = df["mths_since_last_record"].isna().astype(int)
    return df


def calcular_lgd(df):
    """LGD de los creditos castigados.

    EAD (exposicion al incumplimiento) = capital prestado - capital ya pagado.
    Recuperado neto = recoveries - collection_recovery_fee.
    LGD = 1 - recuperado neto / EAD, recortada a [0, 1].
    """
    df = df[df["loan_status"].isin(CASTIGADOS)].copy()
    df["ead"] = df["funded_amnt"] - df["total_rec_prncp"]
    df = df[df["ead"] > 0]
    recuperado = df["recoveries"] - df["collection_recovery_fee"]
    df["lgd"] = (1 - recuperado / df["ead"]).clip(0, 1)
    return df


def procesar(ruta_original, carpeta_salida):
    """Lee el original y guarda pd.parquet y lgd.parquet."""
    carpeta = Path(carpeta_salida)
    carpeta.mkdir(parents=True, exist_ok=True)

    df = leer_original(ruta_original)
    df = variables_derivadas(crear_objetivo(df))
    lgd = calcular_lgd(df)

    columnas_pagos = ["total_rec_prncp", "recoveries", "collection_recovery_fee"]
    df.drop(columns=columnas_pagos).to_parquet(carpeta / "pd.parquet", index=False)
    lgd.to_parquet(carpeta / "lgd.parquet", index=False)
    return df, lgd


def cargar(carpeta, tabla, fraccion=1.0, semilla=0):
    """Lee una tabla procesada ('pd' o 'lgd'); con fraccion < 1 toma una muestra."""
    df = pd.read_parquet(Path(carpeta) / f"{tabla}.parquet")
    if fraccion < 1:
        df = df.sample(frac=fraccion, random_state=semilla)
    return df.reset_index(drop=True)
