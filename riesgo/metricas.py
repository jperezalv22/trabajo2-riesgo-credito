"""Metricas de evaluacion.

Regresion (LGD): MAE y RMSE.
Clasificacion (PD): AUC, Gini, KS y Brier, mas la curva de calibracion por deciles.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

# En estas, un valor mas alto es mejor. En las demas (MAE, RMSE, Brier) es al reves.
MAYOR_ES_MEJOR = {"auc", "gini", "ks"}


def mae(y, y_pred):
    """Error absoluto medio: en promedio, cuanto se equivoca la prediccion."""
    return float(np.mean(np.abs(y - y_pred)))


def rmse(y, y_pred):
    """Raiz del error cuadratico medio: castiga mas los errores grandes."""
    return float(np.sqrt(np.mean((y - y_pred) ** 2)))


def regresion(y, y_pred):
    return {"mae": mae(y, y_pred), "rmse": rmse(y, y_pred)}


def auc(y, p):
    """Area bajo la curva ROC.

    Es la probabilidad de que, al tomar un malo y un bueno al azar, el malo tenga
    la PD mas alta (los empates valen 1/2). 0.5 es azar y 1 es separacion perfecta.
    Solo depende del orden de las PD, no de su valor.
    """
    return float(roc_auc_score(y, p))


def gini(y, p):
    """Indice de Gini = 2 * AUC - 1. Es el AUC en otra escala: 0 es azar y 1 es perfecto."""
    return 2 * auc(y, p) - 1


def ks(y, p):
    """Estadistico KS = maximo de |TPR - FPR| entre todos los umbrales.

    TPR: fraccion de malos con PD por encima del umbral. FPR: lo mismo para los buenos.
    Es la mayor distancia entre las distribuciones acumuladas de PD de malos y de buenos.
    """
    fpr, tpr, _ = roc_curve(y, p, drop_intermediate=False)
    return float(np.max(np.abs(tpr - fpr)))


def brier(y, p):
    """Puntaje de Brier = media de (p - y)^2, con y en {0, 1}. Mas bajo es mejor.

    A diferencia de las anteriores, depende del valor de las PD: castiga a un modelo
    que ordena bien pero dice probabilidades equivocadas.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def entropia_cruzada(y, p):
    """Media de -[y ln(p) + (1 - y) ln(1 - p)]. Es la perdida con que se entrena y se elige."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def clasificacion(y, p):
    return {"auc": auc(y, p), "gini": gini(y, p), "ks": ks(y, p), "brier": brier(y, p)}


def curva_calibracion(y, p, grupos=10):
    """Curva de calibracion: en cada grupo, la PD media predicha contra la tasa real de malos.

    Se ordena por PD y se parte en 'grupos' del mismo tamano (deciles, por defecto).
    Un modelo bien calibrado tiene pd_media parecida a tasa_observada en todos los grupos.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    orden = np.argsort(p, kind="stable")
    filas = []
    for i, indices in enumerate(np.array_split(orden, grupos), start=1):
        filas.append({"grupo": i, "creditos": len(indices),
                      "pd_media": float(p[indices].mean()),
                      "tasa_observada": float(y[indices].mean())})
    return filas


def resumen_semillas(resultados):
    """De una lista de diccionarios de metricas (una por semilla) a media, desv, mejor y peor.

    'mejor' es el valor mas alto en AUC, Gini y KS, y el mas bajo en MAE, RMSE y Brier.
    """
    resumen = {}
    for metrica in resultados[0]:
        valores = np.array([r[metrica] for r in resultados])
        mayor_mejor = metrica in MAYOR_ES_MEJOR
        resumen[metrica] = {"media": valores.mean(), "desv": valores.std(ddof=1),
                            "mejor": valores.max() if mayor_mejor else valores.min(),
                            "peor": valores.min() if mayor_mejor else valores.max()}
    return resumen
