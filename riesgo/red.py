"""Perceptron multicapa y su entrenamiento, comun a la PD y a la LGD.

La salida es una sola neurona con activacion sigmoide, porque las dos cosas que
estimamos estan entre 0 y 1: la probabilidad de incumplir (PD) y la fraccion
que se pierde (LGD).

El bucle de entrenamiento esta escrito a mano para que se vea cada paso:
propagacion hacia adelante, perdida, retropropagacion (loss.backward) y
actualizacion con Adam (optimizador.step).
"""

import copy
import time

import numpy as np
import torch
from torch import nn

ACTIVACIONES = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoide": nn.Sigmoid}


def crear_red(n_entradas, capas, activacion="relu", dropout=0.0):
    """Ejemplo: capas=[64, 32] -> entrada -> 64 -> 32 -> 1 neurona de salida."""
    bloques = []
    anterior = n_entradas
    for n in capas:
        bloques += [nn.Linear(anterior, n), ACTIVACIONES[activacion]()]
        if dropout > 0:
            bloques.append(nn.Dropout(dropout))
        anterior = n
    bloques += [nn.Linear(anterior, 1), nn.Sigmoid()]
    return nn.Sequential(*bloques)


def contar_parametros(red):
    return sum(p.numel() for p in red.parameters() if p.requires_grad)


def funcion_perdida(nombre):
    """bce: entropia cruzada binaria (Bernoulli). mse: error cuadratico (Gauss)."""
    return {"bce": nn.BCELoss(), "mse": nn.MSELoss()}[nombre]


def entrenar(X_ent, y_ent, X_val, y_val, parametros, perdida="bce", semilla=0):
    """Entrena con Adam y parada temprana sobre la perdida de validacion.

    'parametros' es el diccionario de config.yaml (capas, activacion, dropout,
    l2, tasa_aprendizaje, tamano_lote, epocas, paciencia).
    Devuelve la red con los pesos de la mejor epoca y el historial de perdidas.
    """
    torch.manual_seed(semilla)
    generador = np.random.default_rng(semilla)

    red = crear_red(X_ent.shape[1], parametros["capas"], parametros["activacion"],
                    parametros["dropout"])
    criterio = funcion_perdida(perdida)
    # weight_decay en Adam es la penalizacion L2 sobre los pesos
    optimizador = torch.optim.Adam(red.parameters(), lr=parametros["tasa_aprendizaje"],
                                   weight_decay=parametros["l2"])

    X_ent, y_ent = torch.from_numpy(X_ent), torch.from_numpy(y_ent).unsqueeze(1)
    X_val, y_val = torch.from_numpy(X_val), torch.from_numpy(y_val).unsqueeze(1)

    historial = {"entrenamiento": [], "validacion": []}
    mejor = {"perdida": np.inf, "pesos": None, "epoca": 0}
    inicio = time.time()

    for epoca in range(parametros["epocas"]):
        red.train()                      # activa el dropout
        orden = generador.permutation(len(X_ent))
        suma = 0.0
        for i in range(0, len(orden), parametros["tamano_lote"]):
            lote = orden[i:i + parametros["tamano_lote"]]
            optimizador.zero_grad()
            perdida_lote = criterio(red(X_ent[lote]), y_ent[lote])
            perdida_lote.backward()      # retropropagacion: gradientes de todos los pesos
            optimizador.step()           # actualizacion de Adam
            suma += perdida_lote.item() * len(lote)

        red.eval()                       # apaga el dropout para medir
        with torch.no_grad():
            perdida_val = criterio(red(X_val), y_val).item()
        historial["entrenamiento"].append(suma / len(X_ent))
        historial["validacion"].append(perdida_val)

        if perdida_val < mejor["perdida"]:
            mejor = {"perdida": perdida_val, "pesos": copy.deepcopy(red.state_dict()),
                     "epoca": epoca}
        elif epoca - mejor["epoca"] >= parametros["paciencia"]:
            break                        # parada temprana

    red.load_state_dict(mejor["pesos"])
    historial["mejor_epoca"] = mejor["epoca"]
    historial["segundos"] = time.time() - inicio
    return red, historial


def predecir(red, X):
    red.eval()
    with torch.no_grad():
        return red(torch.from_numpy(X)).squeeze(1).numpy()


def guardar(red, parametros, ruta):
    """Guarda la arquitectura y los pesos, para no reentrenar en la app ni en las figuras."""
    n_entradas = red[0].in_features
    torch.save({"n_entradas": n_entradas, "parametros": parametros,
                "pesos": red.state_dict()}, ruta)


def cargar(ruta):
    guardado = torch.load(ruta, weights_only=False)
    p = guardado["parametros"]
    red = crear_red(guardado["n_entradas"], p["capas"], p["activacion"], p["dropout"])
    red.load_state_dict(guardado["pesos"])
    red.eval()
    return red
