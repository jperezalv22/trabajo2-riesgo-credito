"""Parte 1: probabilidad de incumplimiento (PD).

Compara dos modelos con las mismas particiones y el mismo preprocesamiento:
  - regresion logistica: el modelo de referencia, con C elegido en validacion;
  - red neuronal: perceptron multicapa con salida sigmoide, entrenado con
    'pd.semillas' semillas distintas.

Sin pesos por clase ni sobremuestreo: cambian la proporcion de malos que ve el modelo y sus
probabilidades dejarian de estar calibradas, y aqui la probabilidad es lo que se usa.

Salidas: resultados/pd/ (metricas.md, semillas.csv, logistica_c.md, figuras y tablas de
pd_figuras.py) y modelos/ (preprocesador, logistica, red y puntajes, para no reentrenar).

    python pd_entrenar.py                     # completo
    python pd_entrenar.py --demo              # muestra del 10 % y 3 semillas (< 2 min)
    python pd_entrenar.py --demo --set pd.red.capas=[32,16] --set scorecard.pdo=40
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import pd_figuras
from riesgo import configuracion, datos, metricas, particiones, predecir_pd, preparar, red as redes
from riesgo import scorecard


def preparar_pd(cfg, demo):
    """Datos de la PD ya divididos y transformados."""
    fraccion = cfg["datos"]["fraccion_demo"] if demo else 1.0
    df = datos.cargar(cfg["datos"]["procesados"], "pd", fraccion, cfg["semilla"])
    p = cfg["particion"]
    partes = particiones.dividir(df, p["ultimo_anio_entrenamiento"], p["fraccion_validacion"],
                                 cfg["semilla"], estratificar="malo")
    v = cfg["variables"]
    prepro = preparar.crear_preprocesador(v["numericas"], v["categoricas"])
    return partes, prepro, preparar.matrices(prepro, *partes, objetivo="malo")


def ajustar_logistica(m, c):
    """Prueba cada C y se queda con el de menor perdida en VALIDACION (no de prueba)."""
    X_ent, y_ent = m["entrenamiento"]
    X_val, y_val = m["validacion"]
    filas, mejor = [], None
    inicio = time.time()
    for valor_c in c["C"]:
        modelo = LogisticRegression(C=valor_c, max_iter=c["max_iteraciones"]).fit(X_ent, y_ent)
        p_val = modelo.predict_proba(X_val)[:, 1]
        perdida = metricas.entropia_cruzada(y_val, p_val)
        filas.append({"C": valor_c, "Pérdida en validación": round(perdida, 5),
                      "AUC en validación": round(metricas.auc(y_val, p_val), 4),
                      "Iteraciones": int(modelo.n_iter_[0])})
        if mejor is None or perdida < mejor[0]:
            mejor = (perdida, modelo)
    return mejor[1], pd.DataFrame(filas), time.time() - inicio


def entrenar_red_semillas(m, parametros, semillas, semilla_base):
    """Entrena la misma red con varias semillas; devuelve metricas y la mejor red."""
    X_ent, y_ent = m["entrenamiento"]
    X_val, y_val = m["validacion"]
    X_pru, y_pru = m["prueba"]
    filas, mejor = [], None
    for k in range(semillas):
        red, hist = redes.entrenar(X_ent, y_ent, X_val, y_val, parametros, "bce",
                                   semilla=semilla_base + k)
        fila = {"semilla": semilla_base + k, **metricas.clasificacion(y_pru, redes.predecir(red, X_pru)),
                "mejor_epoca": hist["mejor_epoca"], "segundos": hist["segundos"],
                "perdida_validacion": min(hist["validacion"]),
                "auc_validacion": metricas.auc(y_val, redes.predecir(red, X_val))}
        filas.append(fila)
        print(f"  semilla {fila['semilla']}: AUC {fila['auc']:.4f}  KS {fila['ks']:.4f}  "
              f"Brier {fila['brier']:.4f}  ({fila['segundos']:.1f} s)")
        # se guarda la red con mejor perdida de VALIDACION (no de prueba)
        if mejor is None or fila["perdida_validacion"] < mejor[0]:
            mejor = (fila["perdida_validacion"], red)
    return pd.DataFrame(filas), mejor[1]


def tabla_metricas(y_pru, p_logistica, n_c, seg_logistica, n_param_logistica, semillas_df, n_param_red):
    """Metricas en prueba de los dos modelos, en el formato de la tabla del reporte."""
    nombres = {"auc": "AUC", "gini": "Gini", "ks": "KS", "brier": "Brier"}
    r = metricas.clasificacion(y_pru, p_logistica)
    fila_log = {"Modelo": "Regresión logística",
                **{nombres[k]: f"{r[k]:.4f}" for k in nombres},
                "Parámetros": n_param_logistica, "Configuraciones": n_c, "Corridas": 1,
                "Segundos": f"{seg_logistica:.1f}"}
    resumen = metricas.resumen_semillas(semillas_df[list(nombres)].to_dict("records"))
    fila_red = {"Modelo": "Red neuronal",
                **{nombres[k]: f"{s['media']:.4f} ± {s['desv']:.4f} "
                               f"(mejor {s['mejor']:.4f}, peor {s['peor']:.4f})"
                   for k, s in resumen.items()},
                "Parámetros": n_param_red, "Configuraciones": 1, "Corridas": len(semillas_df),
                "Segundos": f"{semillas_df['segundos'].sum():.1f}"}
    return pd.DataFrame([fila_log, fila_red])


def guardar_modelos(carpeta, cfg, prepro, logistica, red, m):
    """Todo lo que la app necesita, para no reentrenar.

    Los puntajes de la poblacion salen del modelo que usa la scorecard, y se guardan junto con
    los parametros que los produjeron para que el percentil nunca mezcle dos escalas distintas.
    """
    parametros = cfg["scorecard"]
    X_ent = m["entrenamiento"][0]
    joblib.dump(prepro, carpeta / "pd_preprocesador.joblib")
    joblib.dump(logistica, carpeta / "pd_logistica.joblib")
    redes.guardar(red, cfg["pd"]["red"], carpeta / "pd_red.pt")
    redes.exportar_numpy(red, cfg["pd"]["red"]["activacion"], carpeta / "pd_red.npz")

    if parametros["modelo"] == "logistica":
        p_ent = logistica.predict_proba(X_ent)[:, 1]
    else:
        p_ent = redes.predecir(red, X_ent)
    np.save(carpeta / "pd_puntajes.npy", scorecard.puntaje(p_ent, parametros).astype(np.float32))

    info = {"scorecard": parametros, "semilla": cfg["semilla"],
            "referencia": [float(v) for v in X_ent.mean(axis=0)]}
    (carpeta / "pd_scorecard.json").write_text(json.dumps(info, indent=1), encoding="utf-8")


def main():
    a = configuracion.argumentos(__doc__).parse_args()
    cfg = configuracion.cargar(a.config, a.cambios)
    c = cfg["pd"]
    semillas = 3 if a.demo else c["semillas"]
    salida = Path("resultados/demo/pd" if a.demo else "resultados/pd")
    carpeta_modelos = Path("resultados/demo/modelos" if a.demo else "modelos")
    salida.mkdir(parents=True, exist_ok=True)
    carpeta_modelos.mkdir(parents=True, exist_ok=True)
    inicio = time.time()

    partes, prepro, m = preparar_pd(cfg, a.demo)
    for fila in particiones.resumen(*partes, objetivo="malo"):
        print(fila)
    X_pru, y_pru = m["prueba"]

    logistica, tabla_c, seg_logistica = ajustar_logistica(m, c["logistica"])
    print(tabla_c.to_markdown(index=False))
    print(f"Mejor C: {logistica.C} ({len(tabla_c)} configuraciones en {seg_logistica:.1f} s)")
    n_param_logistica = logistica.coef_.size + 1

    print(f"Red neuronal ({semillas} semillas):")
    semillas_df, mejor_red = entrenar_red_semillas(m, c["red"], semillas, cfg["semilla"])

    tabla = tabla_metricas(y_pru, logistica.predict_proba(X_pru)[:, 1], len(tabla_c), seg_logistica,
                           n_param_logistica, semillas_df, redes.contar_parametros(mejor_red))
    semillas_df.to_csv(salida / "semillas.csv", index=False)
    (salida / "metricas.md").write_text(tabla.to_markdown(index=False) + "\n", encoding="utf-8")
    (salida / "logistica_c.md").write_text(tabla_c.to_markdown(index=False) + "\n", encoding="utf-8")

    guardar_modelos(carpeta_modelos, cfg, prepro, logistica, mejor_red, m)
    pd_figuras.generar(partes, predecir_pd.cargar(carpeta_modelos), salida)

    print(tabla.to_markdown(index=False))
    print(f"Tiempo total: {time.time() - inicio:.0f} s")


if __name__ == "__main__":
    main()
