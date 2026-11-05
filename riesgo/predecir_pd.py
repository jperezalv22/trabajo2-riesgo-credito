"""Prediccion de la PD para la aplicacion web y la perdida esperada.

Este archivo NO importa torch: la red se predice con riesgo.red_numpy.

    modelos = predecir_pd.cargar("modelos")
    predecir_pd.probabilidad(modelos, df)            # df con las columnas de pd.parquet
    predecir_pd.evaluar_cliente(modelos, cliente)    # cliente es un diccionario

Todo lo que define el puntaje (parametros de la scorecard, modelo, cliente de referencia) se lee
de modelos/pd_scorecard.json y no de config.yaml, porque tiene que ser lo mismo con que se
calcularon los puntajes de la poblacion. Lo escribe pd_entrenar.py.
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from riesgo import scorecard
from riesgo.red_numpy import predecir_numpy

# En estos indicadores, un vacio en la variable de origen significa "nunca ha pasado" (D4).
INDICADORES = {"nunca_en_mora": "meses_desde_mora",
               "sin_registros_publicos": "meses_desde_registro"}


def cargar(carpeta="modelos"):
    """Lee de disco todo lo necesario para predecir."""
    c = Path(carpeta)
    info = json.loads((c / "pd_scorecard.json").read_text(encoding="utf-8"))
    preprocesador = joblib.load(c / "pd_preprocesador.joblib")
    return {"preprocesador": preprocesador,
            "logistica": joblib.load(c / "pd_logistica.joblib"),
            "ruta_red": c / "pd_red.npz",
            "puntajes": np.sort(np.load(c / "pd_puntajes.npy")),
            "grupos": scorecard.grupos_de_columnas(preprocesador),
            "referencia": np.array(info["referencia"], dtype=np.float32),
            "scorecard": info["scorecard"],
            "semilla": info["semilla"]}


def predecir_matriz(modelos, X, modelo=None):
    """Probabilidad de ser malo para una matriz ya transformada."""
    if (modelo or modelos["scorecard"]["modelo"]) == "logistica":
        return modelos["logistica"].predict_proba(X)[:, 1]
    return predecir_numpy(modelos["ruta_red"], X)


def transformar(modelos, df):
    """Mismo preprocesamiento del entrenamiento, en float32 como se entreno."""
    return modelos["preprocesador"].transform(df).astype(np.float32)


def probabilidad(modelos, df, modelo=None):
    """PD de cada fila de df. Por defecto, con el modelo de la scorecard ('red' o 'logistica')."""
    return predecir_matriz(modelos, transformar(modelos, df), modelo)


def fila_cliente(modelos, cliente):
    """De un diccionario a un DataFrame de una fila, con las columnas que espera el modelo.

    Una variable ausente o en None se trata como un faltante en el entrenamiento: las numericas
    se imputan con la mediana y una categoria desconocida queda en ceros. Si falta la clave
    (no el valor) se avisa, porque suele ser un error de digitacion en la app.
    """
    preprocesador = modelos["preprocesador"]
    cliente = dict(cliente)
    for indicador, origen in INDICADORES.items():
        if indicador not in cliente:
            cliente[indicador] = int(pd.isna(cliente.get(origen)))
    # no feature_names_in_: esa lista trae todo el parquet, incluso id y malo, que el modelo no usa
    columnas = list(modelos["grupos"])
    ausentes = [v for v in columnas if v not in cliente]
    if ausentes:
        warnings.warn(f"evaluar_cliente: sin dato para {ausentes}; se imputan.", stacklevel=3)
    fila = pd.DataFrame([cliente]).reindex(columns=columnas)
    for nombre, _, variables in preprocesador.transformers_:
        if nombre in ("log", "num"):
            for v in variables:
                fila[v] = pd.to_numeric(fila[v]).astype(float)
        elif nombre == "cat":
            fila[variables] = fila[variables].astype(object)
    return fila


def evaluar_cliente(modelos, cliente):
    """PD, puntaje, percentil y puntos por variable de una persona.

    Devuelve un diccionario con:
      pd: probabilidad de ser malo.
      puntaje: puntaje en la scorecard (mas alto, menos riesgo).
      percentil: % de la poblacion de entrenamiento con un puntaje menor o igual.
      puntos: {variable: puntos}, cuanto sube o baja el puntaje esa variable frente al cliente
        de referencia (el promedio de entrenamiento).
      puntos_base: puntaje del cliente de referencia. Se cumple que
        puntos_base + suma(puntos) = puntaje.
    """
    x = transformar(modelos, fila_cliente(modelos, cliente))[0]
    p = float(predecir_matriz(modelos, x[None, :])[0])
    parametros = modelos["scorecard"]
    s = float(scorecard.puntaje(p, parametros))
    referencia = modelos["referencia"]
    base = float(scorecard.puntaje(predecir_matriz(modelos, referencia[None, :])[0], parametros))

    if parametros["modelo"] == "logistica":
        puntos = scorecard.puntos_logistica(modelos["logistica"], x, referencia, modelos["grupos"],
                                            parametros)
    else:
        puntos = scorecard.puntos_permutacion(lambda X: predecir_matriz(modelos, X), x, referencia,
                                              modelos["grupos"], parametros,
                                              parametros["permutaciones"], modelos["semilla"])

    poblacion = modelos["puntajes"]
    percentil = 100 * np.searchsorted(poblacion, s, side="right") / len(poblacion)
    return {"pd": p, "puntaje": s, "percentil": float(percentil), "puntos": puntos,
            "puntos_base": base}
