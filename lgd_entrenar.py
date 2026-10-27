"""Parte 2: perdida dado incumplimiento (LGD).

Compara tres modelos con las mismas particiones y el mismo preprocesamiento:
  - media: siempre predice la LGD promedio de entrenamiento (piso de referencia);
  - regresion lineal: el modelo de baja complejidad;
  - red neuronal: perceptron multicapa con salida sigmoide, entrenado con
    'lgd.semillas' semillas distintas.

Salidas: resultados/lgd/ (metricas.md, semillas.csv, distribucion.png, cuantiles.md)
y modelos/ (red, regresion lineal y preprocesador, para no reentrenar).

    python lgd_entrenar.py                     # completo
    python lgd_entrenar.py --demo              # muestra del 10 % y 3 semillas (< 2 min)
    python lgd_entrenar.py --set lgd.perdida=mse --set lgd.red.capas=[64,32]
"""

import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from riesgo import configuracion, datos, metricas, particiones, preparar, red as redes


def preparar_lgd(cfg, demo):
    """Datos de la LGD ya divididos y transformados."""
    c = cfg["lgd"]
    fraccion = cfg["datos"]["fraccion_demo"] if demo else 1.0
    df = datos.cargar(cfg["datos"]["procesados"], "lgd", fraccion, cfg["semilla"])
    df = df[df["anio"] <= c["ultimo_anio_datos"]]
    partes = particiones.dividir(df, c["ultimo_anio_entrenamiento"], c["fraccion_validacion"],
                                 cfg["semilla"])
    v = cfg["variables"]
    prepro = preparar.crear_preprocesador(v["numericas"], v["categoricas"])
    return partes, prepro, preparar.matrices(prepro, *partes, objetivo="lgd")


def entrenar_red_semillas(m, parametros, perdida, semillas, semilla_base):
    """Entrena la misma red con varias semillas; devuelve metricas y la mejor red."""
    X_ent, y_ent = m["entrenamiento"]
    X_val, y_val = m["validacion"]
    X_pru, y_pru = m["prueba"]
    filas, mejor = [], None
    for k in range(semillas):
        red, hist = redes.entrenar(X_ent, y_ent, X_val, y_val, parametros, perdida,
                                   semilla=semilla_base + k)
        pred = redes.predecir(red, X_pru)
        fila = {"semilla": semilla_base + k, **metricas.regresion(y_pru, pred),
                "mejor_epoca": hist["mejor_epoca"], "segundos": hist["segundos"],
                "perdida_validacion": min(hist["validacion"])}
        filas.append(fila)
        print(f"  semilla {fila['semilla']}: MAE {fila['mae']:.4f}  RMSE {fila['rmse']:.4f}  "
              f"({fila['segundos']:.1f} s)")
        # se guarda la red con mejor perdida de VALIDACION (no de prueba)
        if mejor is None or fila["perdida_validacion"] < mejor[0]:
            mejor = (fila["perdida_validacion"], red, pred)
    return pd.DataFrame(filas), mejor[1], mejor[2]


def tabla_metricas(y_pru, pred_media, pred_lineal, semillas_df, n_param_lineal, n_param_red,
                   seg_lineal):
    filas = []
    for nombre, pred, n_param, seg in [("Media (constante)", pred_media, 1, 0.0),
                                       ("Regresión lineal", pred_lineal, n_param_lineal, seg_lineal)]:
        r = metricas.regresion(y_pru, pred)
        filas.append({"Modelo": nombre, "MAE": f"{r['mae']:.4f}", "RMSE": f"{r['rmse']:.4f}",
                      "Parámetros": n_param, "Corridas": 1, "Segundos": f"{seg:.1f}"})
    for metrica in ["mae", "rmse"]:
        s = semillas_df[metrica]
        texto = f"{s.mean():.4f} ± {s.std(ddof=1):.4f} (mejor {s.min():.4f}, peor {s.max():.4f})"
        if metrica == "mae":
            fila = {"Modelo": "Red neuronal", "MAE": texto}
        else:
            fila["RMSE"] = texto
    fila.update({"Parámetros": n_param_red, "Corridas": len(semillas_df),
                 "Segundos": f"{semillas_df['segundos'].sum():.1f}"})
    filas.append(fila)
    return pd.DataFrame(filas)


def cuantiles(y_pru, predicciones):
    """Compara la distribucion observada con la predicha en algunos puntos."""
    puntos = [0.10, 0.25, 0.50, 0.75, 0.90]
    filas = []
    for nombre, v in [("Observada", y_pru), *predicciones.items()]:
        fila = {"Distribución": nombre}
        fila.update({f"p{int(q * 100)}": f"{np.quantile(v, q):.3f}" for q in puntos})
        fila["Fuera de [0, 1]"] = f"{np.mean((v < 0) | (v > 1)):.1%}"
        fila["≥ 0.99"] = f"{np.mean(v >= 0.99):.1%}"
        filas.append(fila)
    return pd.DataFrame(filas)


def figura_distribucion(y_pru, predicciones, ruta):
    fig, eje = plt.subplots(figsize=(8, 4.5))
    bordes = np.linspace(min(0, *(p.min() for p in predicciones.values())), 1.05, 60)
    eje.hist(y_pru, bins=bordes, alpha=0.45, label="Observada", color="#23201c", density=True)
    for (nombre, pred), color in zip(predicciones.items(), ["#b4532a", "#2f6f4f"]):
        eje.hist(pred, bins=bordes, histtype="step", linewidth=2, label=nombre,
                 color=color, density=True)
    eje.set_xlabel("LGD")
    eje.set_ylabel("densidad")
    eje.legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def main():
    a = configuracion.argumentos(__doc__).parse_args()
    cfg = configuracion.cargar(a.config, a.cambios)
    c = cfg["lgd"]
    semillas = 3 if a.demo else c["semillas"]
    salida = Path("resultados/demo/lgd" if a.demo else "resultados/lgd")
    carpeta_modelos = Path("resultados/demo/modelos" if a.demo else "modelos")
    salida.mkdir(parents=True, exist_ok=True)
    carpeta_modelos.mkdir(parents=True, exist_ok=True)

    partes, prepro, m = preparar_lgd(cfg, a.demo)
    for fila in particiones.resumen(*partes, objetivo="lgd"):
        print(fila)
    X_ent, y_ent = m["entrenamiento"]
    X_pru, y_pru = m["prueba"]

    pred_media = np.full_like(y_pru, y_ent.mean())

    inicio = time.time()
    lineal = LinearRegression().fit(X_ent, y_ent)
    seg_lineal = time.time() - inicio
    pred_lineal = lineal.predict(X_pru)
    n_param_lineal = lineal.coef_.size + 1

    print(f"Red neuronal ({semillas} semillas, pérdida {c['perdida']}):")
    semillas_df, mejor_red, pred_red = entrenar_red_semillas(
        m, c["red"], c["perdida"], semillas, cfg["semilla"])

    tabla = tabla_metricas(y_pru, pred_media, pred_lineal, semillas_df, n_param_lineal,
                           redes.contar_parametros(mejor_red), seg_lineal)
    predicciones = {"Regresión lineal": pred_lineal, "Red neuronal": pred_red}
    tabla_q = cuantiles(y_pru, predicciones)

    semillas_df.to_csv(salida / "semillas.csv", index=False)
    (salida / "metricas.md").write_text(tabla.to_markdown(index=False) + "\n", encoding="utf-8")
    (salida / "cuantiles.md").write_text(tabla_q.to_markdown(index=False) + "\n", encoding="utf-8")
    figura_distribucion(y_pru, predicciones, salida / "distribucion.png")

    redes.guardar(mejor_red, c["red"], carpeta_modelos / "lgd_red.pt")
    joblib.dump(lineal, carpeta_modelos / "lgd_lineal.joblib")
    joblib.dump(prepro, carpeta_modelos / "lgd_preprocesador.joblib")

    print(tabla.to_markdown(index=False))
    print(tabla_q.to_markdown(index=False))


if __name__ == "__main__":
    main()
