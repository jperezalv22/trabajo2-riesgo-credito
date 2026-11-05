"""Figuras y tablas de la PD a partir de los modelos guardados: no reentrena ni usa PyTorch.

Salidas en resultados/pd/: roc.png, calibracion.png, puntaje.png, por_particion.md,
calibracion.md, calibracion.csv y puntaje_bandas.md. pd_entrenar.py lo llama al terminar, y se
puede correr solo para regenerar las figuras.

    python pd_figuras.py
    python pd_figuras.py --demo     # con los modelos guardados y una muestra del 10 %
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from riesgo import configuracion, datos, metricas, particiones, predecir_pd, scorecard

# Azul y no el verde de la figura de la LGD: terracota y verde casi no se distinguen con
# daltonismo protan (dE 4.6, cuando el minimo aceptable es 8); terracota y azul, dE 20.
COLORES = {"Regresión logística": "#b4532a", "Red neuronal": "#2b6cb0"}
MARCADORES = {"Regresión logística": "o", "Red neuronal": "s"}
CLAVES = {"Regresión logística": "logistica", "Red neuronal": "red"}
TINTA, TINTA_SUAVE, GRIS, GRIS_RELLENO = "#23201c", "#6b655c", "#c9c3b6", "#8f897c"


def partes_evaluacion(cfg, demo):
    """Las mismas particiones del entrenamiento completo; en demo, una muestra de cada una.

    Se divide primero y se muestrea despues para que ningun credito de entrenamiento termine
    evaluandose como si fuera de validacion.
    """
    df = datos.cargar(cfg["datos"]["procesados"], "pd")
    p = cfg["particion"]
    partes = particiones.dividir(df, p["ultimo_anio_entrenamiento"], p["fraccion_validacion"],
                                 cfg["semilla"], estratificar="malo")
    if demo:
        partes = [x.sample(frac=cfg["datos"]["fraccion_demo"], random_state=cfg["semilla"])
                  for x in partes]
    return partes


def ejes_limpios(eje):
    """Quita lo que no aporta: la tinta queda en los datos."""
    eje.spines[["top", "right"]].set_visible(False)
    eje.spines[["left", "bottom"]].set_color(GRIS)
    eje.tick_params(colors=TINTA_SUAVE, labelsize=9)
    eje.grid(color=GRIS, linewidth=0.5, alpha=0.5)
    eje.set_axisbelow(True)


def figura_roc(y, predicciones, ruta):
    fig, eje = plt.subplots(figsize=(5.5, 5.5))
    eje.plot([0, 1], [0, 1], color=GRIS, linewidth=1.5, linestyle="--", label="Azar (AUC = 0.5)")
    for nombre, p in predicciones.items():
        fpr, tpr, _ = roc_curve(y, p)
        eje.plot(fpr, tpr, color=COLORES[nombre], linewidth=2,
                 label=f"{nombre} (AUC = {metricas.auc(y, p):.3f})")
    eje.set_xlabel("Tasa de falsos positivos (buenos rechazados)")
    eje.set_ylabel("Tasa de verdaderos positivos (malos detectados)")
    eje.set_xlim(0, 1)
    eje.set_ylim(0, 1)
    eje.set_aspect("equal")
    eje.legend(frameon=False, loc="lower right", fontsize=9)
    ejes_limpios(eje)
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def figura_calibracion(curvas, ruta):
    """Un panel por particion, con la misma escala: se compara validacion contra prueba."""
    fig, ejes = plt.subplots(1, len(curvas), figsize=(10, 5), sharex=True, sharey=True)
    tope = 1.05 * max(max(f["pd_media"], f["tasa_observada"])
                      for por_modelo in curvas.values()
                      for filas in por_modelo.values() for f in filas)
    for eje, (particion, por_modelo) in zip(ejes, curvas.items()):
        eje.plot([0, tope], [0, tope], color=GRIS, linewidth=1.5, linestyle="--",
                 label="Calibración perfecta")
        for nombre, filas in por_modelo.items():
            eje.plot([f["pd_media"] for f in filas], [f["tasa_observada"] for f in filas],
                     marker=MARCADORES[nombre], color=COLORES[nombre], linewidth=2, markersize=6,
                     label=nombre)
        eje.set_title(particion, fontsize=11, color=TINTA)
        eje.set_xlabel("PD media predicha en cada decil")
        eje.set_xlim(0, tope)
        eje.set_ylim(0, tope)
        eje.set_aspect("equal")
        ejes_limpios(eje)
    ejes[0].set_ylabel("Tasa de malos observada en cada decil")
    ejes[0].legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def figura_puntaje(puntajes, y, nombre_modelo, ruta):
    """Distribucion del puntaje de buenos y malos: entre mas separadas, mejor discrimina."""
    fig, eje = plt.subplots(figsize=(8, 4.5))
    bordes = np.linspace(np.quantile(puntajes, 0.001), np.quantile(puntajes, 0.999), 50)
    eje.hist(puntajes[y == 0], bins=bordes, density=True, color=GRIS_RELLENO, label="Buenos (pagaron)")
    eje.hist(puntajes[y == 1], bins=bordes, density=True, histtype="step", linewidth=2,
             color=TINTA, label="Malos (incumplieron)")
    eje.set_xlabel(f"Puntaje de la scorecard ({nombre_modelo}); más alto es menos riesgo")
    eje.set_ylabel("Densidad")
    eje.legend(frameon=False, fontsize=9)
    ejes_limpios(eje)
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def tabla_por_particion(partes_con_nombre, predicciones):
    """Las mismas metricas donde hay y donde no hay sesgo de madurez (HANDOFF D5)."""
    filas = []
    for particion, (df, por_modelo) in zip(partes_con_nombre, predicciones):
        y = df["malo"].to_numpy()
        for nombre, p in por_modelo.items():
            r = metricas.clasificacion(y, p)
            filas.append({"Partición": particion, "Modelo": nombre, "Créditos": f"{len(y):,}".replace(",", " "),
                          "AUC": f"{r['auc']:.4f}", "Gini": f"{r['gini']:.4f}", "KS": f"{r['ks']:.4f}",
                          "Brier": f"{r['brier']:.4f}", "PD media predicha": f"{p.mean():.1%}",
                          "Tasa de malos": f"{y.mean():.1%}"})
    return pd.DataFrame(filas)


def tabla_bandas(partes, modelos):
    """Tasa de malos por banda de puntaje, contra la PD que promete la formula."""
    parametros = modelos["scorecard"]
    base, pdo = parametros["puntaje_base"], parametros["pdo"]
    cortes = base + pdo * np.arange(-3, 4)
    bordes = [-np.inf, *cortes, np.inf]
    filas = [{"Banda": f"< {cortes[0]:.0f}" if np.isinf(a) else
              f">= {a:.0f}" if np.isinf(b) else f"{a:.0f} a {b:.0f}"}
             for a, b in zip(bordes, bordes[1:])]
    for particion, df in partes.items():
        s = scorecard.puntaje(predecir_pd.probabilidad(modelos, df), parametros)
        y = df["malo"].to_numpy()
        banda = np.digitize(s, cortes)
        for i, fila in enumerate(filas):
            dentro = banda == i
            fila[f"% de créditos ({particion})"] = f"{dentro.mean():.1%}"
            fila[f"PD predicha ({particion})"] = (
                f"{scorecard.probabilidad(s[dentro], parametros).mean():.1%}" if dentro.any() else "-")
            fila[f"Tasa de malos ({particion})"] = f"{y[dentro].mean():.1%}" if dentro.any() else "-"
    return pd.DataFrame(filas)


def generar(partes, modelos, salida):
    """Todas las figuras y tablas, con los modelos ya entrenados."""
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    _, validacion, prueba = partes
    grupos = {"Validación (hasta 2013)": validacion, "Prueba (2014-2015)": prueba,
              "Prueba, solo 2014": prueba[prueba["anio"] == 2014],
              "Prueba, solo 2015": prueba[prueba["anio"] == 2015]}
    predicciones = {particion: {nombre: predecir_pd.probabilidad(modelos, df, clave)
                                for nombre, clave in CLAVES.items()}
                    for particion, df in grupos.items()}

    por_particion = tabla_por_particion(
        grupos, [(df, predicciones[particion]) for particion, df in grupos.items()])
    (salida / "por_particion.md").write_text(por_particion.to_markdown(index=False) + "\n", encoding="utf-8")

    principales = ["Validación (hasta 2013)", "Prueba (2014-2015)"]
    curvas = {particion: {nombre: metricas.curva_calibracion(grupos[particion]["malo"].to_numpy(), p)
                          for nombre, p in predicciones[particion].items()}
              for particion in principales}
    filas_csv = [{"particion": particion, "modelo": nombre, **f}
                 for particion, por_modelo in curvas.items()
                 for nombre, filas in por_modelo.items() for f in filas]
    calibracion = pd.DataFrame(filas_csv)
    calibracion.to_csv(salida / "calibracion.csv", index=False)
    (salida / "calibracion.md").write_text(_tabla_calibracion(calibracion) + "\n", encoding="utf-8")

    bandas = tabla_bandas({"validación": validacion, "prueba": prueba}, modelos)
    (salida / "puntaje_bandas.md").write_text(bandas.to_markdown(index=False) + "\n", encoding="utf-8")

    y_prueba = prueba["malo"].to_numpy()
    figura_roc(y_prueba, predicciones["Prueba (2014-2015)"], salida / "roc.png")
    figura_calibracion(curvas, salida / "calibracion.png")
    modelo_scorecard = {v: k for k, v in CLAVES.items()}[modelos["scorecard"]["modelo"]]
    puntajes = scorecard.puntaje(predicciones["Prueba (2014-2015)"][modelo_scorecard], modelos["scorecard"])
    figura_puntaje(puntajes, y_prueba, modelo_scorecard.lower(), salida / "puntaje.png")
    print(por_particion.to_markdown(index=False))


def _tabla_calibracion(calibracion):
    """Una fila por decil, con validacion y prueba lado a lado, para los dos modelos."""
    t = calibracion.assign(columna=calibracion["modelo"] + ", " + calibracion["particion"].str.split(" ").str[0])
    pd_media = t.pivot(index="grupo", columns="columna", values="pd_media")
    tasa = t.pivot(index="grupo", columns="columna", values="tasa_observada")
    salida = pd.DataFrame({"Decil": pd_media.index})
    for columna in pd_media.columns:
        salida[columna] = [f"{a:.1%} / {b:.1%}" for a, b in zip(pd_media[columna], tasa[columna])]
    return salida.to_markdown(index=False)


def main():
    a = configuracion.argumentos(__doc__).parse_args()
    cfg = configuracion.cargar(a.config, a.cambios)
    salida = "resultados/demo/pd" if a.demo else "resultados/pd"
    generar(partes_evaluacion(cfg, a.demo), predecir_pd.cargar("modelos"), salida)


if __name__ == "__main__":
    main()
