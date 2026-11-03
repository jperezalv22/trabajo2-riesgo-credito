"""Prediccion de una red ya entrenada usando solo NumPy, para la aplicacion web.

La app solo necesita predecir, y PyTorch pesa cientos de MB en el servidor.
Una red entrenada es solo multiplicar matrices y aplicar activaciones:
    h = activacion(h @ W.T + b) en cada capa oculta, y sigmoide en la salida.

Los pesos se exportan al entrenar con riesgo.red.exportar_numpy (archivo .npz).
Este archivo NO importa torch: la app debe importar desde aqui, no desde riesgo.red.
"""

import numpy as np

FUNCIONES = {
    "relu": lambda z: np.maximum(z, 0),
    "tanh": np.tanh,
    "sigmoide": lambda z: 1 / (1 + np.exp(-z)),
}


def predecir_numpy(ruta_npz, X):
    """Lo mismo que riesgo.red.predecir(), pero con los pesos exportados."""
    pesos = np.load(ruta_npz)
    activacion = FUNCIONES[str(pesos["activacion"])]
    n = int(pesos["n_capas"])
    h = np.asarray(X, dtype=np.float32)
    for i in range(n):
        z = h @ pesos[f"W{i}"].T + pesos[f"b{i}"]
        h = activacion(z) if i < n - 1 else FUNCIONES["sigmoide"](z)
    return h[:, 0]
