"""Preprocesamiento comun a todos los modelos.

Se ajusta SOLO con el conjunto de entrenamiento y despues se aplica igual a
validacion, prueba y a la aplicacion web. Asi todos los modelos de una tarea
usan exactamente las mismas entradas (regla de comparacion justa).

- numericas: los faltantes se llenan con la mediana y despues se estandariza
  (media 0, desviacion 1). Ingresos y saldos se pasan antes por log(1 + x),
  porque tienen colas muy largas.
- categoricas: one-hot. Categorias que no se vieron en entrenamiento quedan en ceros.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# variables con colas largas (unos pocos clientes con valores enormes)
CON_LOGARITMO = ["annual_inc", "revol_bal", "loan_amnt"]


def crear_preprocesador(numericas, categoricas):
    con_log = [v for v in numericas if v in CON_LOGARITMO]
    sin_log = [v for v in numericas if v not in CON_LOGARITMO]
    return ColumnTransformer([
        ("log", make_pipeline(SimpleImputer(strategy="median"),
                              FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                              StandardScaler()), con_log),
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), sin_log),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50,
                              sparse_output=False), categoricas),
    ])


def matrices(preprocesador, entrenamiento, validacion, prueba, objetivo):
    """Ajusta con entrenamiento y transforma las tres particiones.

    Devuelve un diccionario {'entrenamiento': (X, y), 'validacion': ..., 'prueba': ...}
    con X como float32, que es lo que usa PyTorch.
    """
    X_ent = preprocesador.fit_transform(entrenamiento)
    salida = {"entrenamiento": (X_ent, entrenamiento[objetivo].to_numpy())}
    for nombre, parte in [("validacion", validacion), ("prueba", prueba)]:
        salida[nombre] = (preprocesador.transform(parte), parte[objetivo].to_numpy())
    return {k: (X.astype(np.float32), y.astype(np.float32)) for k, (X, y) in salida.items()}
