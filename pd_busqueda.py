"""Busqueda aleatoria de hiperparametros de la red de la PD.

Se prueban 'pd.busqueda.configuraciones' combinaciones al azar de la rejilla de config.yaml y
se elige la de menor perdida en VALIDACION (la prueba no se toca). Usamos busqueda aleatoria
porque, con el mismo numero de intentos, explora mas valores de cada hiperparametro que una
rejilla completa (Bergstra y Bengio, 2012).

Salida: resultados/pd/busqueda.csv y busqueda.md (la tabla para el reporte, con el tiempo de
cada configuracion). La mejor configuracion se copia a mano a pd.red en config.yaml (asi queda
a la vista que se cambio).

    python pd_busqueda.py
    python pd_busqueda.py --demo              # muestra del 10 % y 6 configuraciones (< 2 min)
"""

import time
from pathlib import Path

import pandas as pd

from lgd_busqueda import sortear
from pd_entrenar import preparar_pd
from riesgo import configuracion, metricas, red as redes


def tabla_reporte(tabla):
    """La busqueda ordenada por perdida de validacion, con nombres para el reporte."""
    t = tabla.rename(columns={
        "configuracion": "Intento", "capas": "Capas ocultas", "activacion": "Activación",
        "dropout": "Dropout", "l2": "L2", "tasa_aprendizaje": "Tasa de aprendizaje",
        "tamano_lote": "Lote", "optimizador": "Optimizador",
        "perdida_validacion": "Pérdida en validación", "auc_validacion": "AUC en validación",
        "parametros_entrenables": "Parámetros", "mejor_epoca": "Mejor época",
        "segundos": "Segundos"})
    t["Pérdida en validación"] = t["Pérdida en validación"].map("{:.5f}".format)
    t["AUC en validación"] = t["AUC en validación"].map("{:.4f}".format)
    return t.to_markdown(index=False)


def main():
    a = configuracion.argumentos(__doc__).parse_args()
    cfg = configuracion.cargar(a.config, a.cambios)
    c = cfg["pd"]
    rejilla = {k: v for k, v in c["busqueda"].items() if k != "configuraciones"}
    n = 6 if a.demo else c["busqueda"]["configuraciones"]
    _, _, m = preparar_pd(cfg, a.demo)
    X_ent, y_ent = m["entrenamiento"]
    X_val, y_val = m["validacion"]

    filas = []
    inicio = time.time()
    for i, combo in enumerate(sortear(rejilla, n, cfg["semilla"])):
        parametros = {**c["red"], **combo}
        red, hist = redes.entrenar(X_ent, y_ent, X_val, y_val, parametros, "bce",
                                   semilla=cfg["semilla"])
        fila = {"configuracion": i + 1, **combo,
                "perdida_validacion": min(hist["validacion"]),
                "auc_validacion": metricas.auc(y_val, redes.predecir(red, X_val)),
                "parametros_entrenables": redes.contar_parametros(red),
                "mejor_epoca": hist["mejor_epoca"],
                "segundos": round(hist["segundos"], 1)}
        filas.append(fila)
        print(f"{i + 1:2d} {combo}  perdida val {fila['perdida_validacion']:.5f}  "
              f"AUC val {fila['auc_validacion']:.4f}  ({fila['segundos']} s)")

    tabla = pd.DataFrame(filas).sort_values("perdida_validacion")
    salida = Path("resultados/demo/pd" if a.demo else "resultados/pd")
    salida.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(salida / "busqueda.csv", index=False)
    (salida / "busqueda.md").write_text(tabla_reporte(tabla) + "\n", encoding="utf-8")
    print(f"\n{len(filas)} configuraciones en {time.time() - inicio:.0f} s. Mejor:")
    print(tabla.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
