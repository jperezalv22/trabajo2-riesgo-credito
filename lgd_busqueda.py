"""Busqueda aleatoria de hiperparametros de la red de la LGD.

Se prueban 'lgd.busqueda.configuraciones' combinaciones al azar de la rejilla de
config.yaml y se elige la de menor perdida en VALIDACION (la prueba no se toca).
Usamos busqueda aleatoria porque, con el mismo numero de intentos, explora mas
valores de cada hiperparametro que una rejilla completa (Bergstra y Bengio, 2012).

Salida: resultados/lgd/busqueda.csv, con el tiempo total. La mejor configuracion
se copia a mano a lgd.red en config.yaml (asi queda a la vista que se cambio).

    python lgd_busqueda.py
    python lgd_busqueda.py --set lgd.busqueda.configuraciones=4 --demo
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from lgd_entrenar import preparar_lgd
from riesgo import configuracion, metricas, red as redes


def sortear(rejilla, n, semilla):
    """n combinaciones al azar (sin repetir) de los valores de la rejilla."""
    generador = np.random.default_rng(semilla)
    vistas, combinaciones = set(), []
    while len(combinaciones) < n:
        combo = {k: v[generador.integers(len(v))] for k, v in rejilla.items()}
        clave = str(combo)
        if clave not in vistas:
            vistas.add(clave)
            combinaciones.append(combo)
    return combinaciones


def main():
    a = configuracion.argumentos(__doc__).parse_args()
    cfg = configuracion.cargar(a.config, a.cambios)
    c = cfg["lgd"]
    rejilla = {k: v for k, v in c["busqueda"].items() if k != "configuraciones"}
    _, _, m = preparar_lgd(cfg, a.demo)
    X_ent, y_ent = m["entrenamiento"]
    X_val, y_val = m["validacion"]

    filas = []
    inicio = time.time()
    for i, combo in enumerate(sortear(rejilla, c["busqueda"]["configuraciones"], cfg["semilla"])):
        parametros = {**c["red"], **combo}
        red, hist = redes.entrenar(X_ent, y_ent, X_val, y_val, parametros, c["perdida"],
                                   semilla=cfg["semilla"])
        pred = redes.predecir(red, X_val)
        fila = {"configuracion": i + 1, **combo,
                "perdida_validacion": min(hist["validacion"]),
                "mae_validacion": metricas.mae(y_val, pred),
                "parametros_entrenables": redes.contar_parametros(red),
                "segundos": round(hist["segundos"], 1)}
        filas.append(fila)
        print(f"{i + 1:2d} {combo}  MAE val {fila['mae_validacion']:.4f}")

    tabla = pd.DataFrame(filas).sort_values("perdida_validacion")
    salida = Path("resultados/demo/lgd" if a.demo else "resultados/lgd")
    salida.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(salida / "busqueda.csv", index=False)
    print(f"\n{len(filas)} configuraciones en {time.time() - inicio:.0f} s. Mejor:")
    print(tabla.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
