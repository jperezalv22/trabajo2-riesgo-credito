"""Scorecard: de la probabilidad de ser malo a un puntaje, y del puntaje a puntos por variable.

Derivacion. Con odds = (1 - p) / p (buenos por cada malo) se pide un puntaje lineal en ln(odds):

    puntaje = offset + factor * ln(odds)

y dos condiciones:
    1. con odds = odds_referencia, el puntaje es puntaje_base;
    2. si las odds se duplican, el puntaje sube pdo puntos.

Al restar las dos ecuaciones: pdo = factor * ln(2). Por eso:

    factor = pdo / ln(2)
    offset = puntaje_base - factor * ln(odds_referencia)

Mas puntaje es menos riesgo. Como solo se cambia de escala, el orden de los clientes no cambia:
base, odds y PDO no mueven ni el AUC ni el percentil de nadie.

'parametros' es siempre la seccion scorecard de config.yaml.
"""

import numpy as np

# Evita ln(0) con PD de exactamente 0 o 1 (la red trabaja en float32).
P_MINIMA = 1e-7


def factor(parametros):
    return parametros["pdo"] / np.log(2)


def offset(parametros):
    return parametros["puntaje_base"] - factor(parametros) * np.log(parametros["odds_referencia"])


def puntaje(p, parametros):
    """De probabilidad de ser malo a puntaje."""
    p = np.clip(np.asarray(p, dtype=float), P_MINIMA, 1 - P_MINIMA)
    return offset(parametros) + factor(parametros) * np.log((1 - p) / p)


def probabilidad(s, parametros):
    """De puntaje a probabilidad de ser malo (la inversa de puntaje)."""
    odds = np.exp((np.asarray(s, dtype=float) - offset(parametros)) / factor(parametros))
    return 1 / (1 + odds)


def grupos_de_columnas(preprocesador):
    """Que columnas de la matriz transformada salen de cada variable original.

    El one-hot convierte una variable en varias columnas, pero la persona tiene un solo
    valor de esa variable: los puntos se reparten por variable, no por columna.
    """
    grupos = {}
    for nombre, _, variables in preprocesador.transformers_:
        if nombre == "remainder":
            continue
        for variable in variables:
            grupos[variable] = []
    for i, columna in enumerate(preprocesador.get_feature_names_out()):
        sin_prefijo = columna.split("__", 1)[1]
        # las numericas se llaman igual que la variable; las one-hot, variable_categoria
        variable = sin_prefijo if sin_prefijo in grupos else next(
            v for v in grupos if sin_prefijo.startswith(f"{v}_"))
        grupos[variable].append(i)
    return grupos


def puntos_logistica(logistica, x, referencia, grupos, parametros):
    """Puntos de cada variable con la regresion logistica, directos de los coeficientes.

    El modelo es ln(p / (1 - p)) = b0 + suma(b_j x_j), y ln(odds) es exactamente lo contrario.
    Por eso, frente al cliente de referencia, cada variable aporta
        puntos = -factor * suma de b_j (x_j - referencia_j) en sus columnas.
    Un coeficiente positivo (mas riesgo) resta puntos.
    """
    b = logistica.coef_[0]
    f = factor(parametros)
    return {v: float(-f * b[c] @ (x[c] - referencia[c])) for v, c in grupos.items()}


def puntos_permutacion(predecir, x, referencia, grupos, parametros, permutaciones, semilla):
    """Puntos de cada variable con cualquier modelo: valores de Shapley por permutaciones.

    Se parte del cliente de referencia y se le cambian sus variables por las de la persona, una
    a la vez y en orden aleatorio. Lo que sube o baja el puntaje en cada cambio se le acredita a
    esa variable, y se promedia entre muchos ordenes (Strumbelj y Kononenko, 2014). Se promedia
    porque en una red el aporte de una variable depende de cuales ya se cambiaron.

    Cada orden suma exactamente puntaje(x) - puntaje(referencia), asi que la suma de los puntos
    cuadra siempre con el puntaje. Es determinista: misma persona y semilla, mismos puntos.

    'predecir' recibe una matriz y devuelve probabilidades.
    """
    nombres = list(grupos)
    d = len(nombres)
    generador = np.random.default_rng(semilla)
    ordenes = [generador.permutation(d) for _ in range(permutaciones)]

    # cada camino pasa por d + 1 clientes: la referencia y un cambio por variable
    X = np.empty((permutaciones * (d + 1), len(x)), dtype=np.float32)
    fila = 0
    for orden in ordenes:
        actual = referencia.copy()
        X[fila] = actual
        fila += 1
        for j in orden:
            columnas = grupos[nombres[j]]
            actual[columnas] = x[columnas]
            X[fila] = actual
            fila += 1

    s = puntaje(predecir(X), parametros).reshape(permutaciones, d + 1)
    cambios = np.diff(s, axis=1)
    puntos = np.zeros(d)
    for i, orden in enumerate(ordenes):
        puntos[orden] += cambios[i]
    return {v: float(valor) for v, valor in zip(nombres, puntos / permutaciones)}
