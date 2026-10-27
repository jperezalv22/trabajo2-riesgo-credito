"""Metricas de evaluacion.

Regresion (LGD): MAE y RMSE.
Las de clasificacion (AUC, Gini, KS, Brier) van en este mismo archivo (parte de la PD).
"""

import numpy as np


def mae(y, y_pred):
    """Error absoluto medio: en promedio, cuanto se equivoca la prediccion."""
    return float(np.mean(np.abs(y - y_pred)))


def rmse(y, y_pred):
    """Raiz del error cuadratico medio: castiga mas los errores grandes."""
    return float(np.sqrt(np.mean((y - y_pred) ** 2)))


def regresion(y, y_pred):
    return {"mae": mae(y, y_pred), "rmse": rmse(y, y_pred)}


def resumen_semillas(resultados):
    """De una lista de diccionarios de metricas (una por semilla) a media, desv, mejor y peor.

    Para MAE y RMSE, 'mejor' es el valor mas bajo.
    """
    resumen = {}
    for metrica in resultados[0]:
        valores = np.array([r[metrica] for r in resultados])
        resumen[metrica] = {"media": valores.mean(), "desv": valores.std(ddof=1),
                            "mejor": valores.min(), "peor": valores.max()}
    return resumen
