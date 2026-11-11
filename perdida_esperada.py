"""Parte 2, punto 5: perdida esperada de cada credito, E[L] = PD * LGD * EAD.

Junta los dos modelos ya entrenados, sin reentrenar nada (no usa PyTorch):
  - PD: el modelo de la scorecard (riesgo/predecir_pd.py), red por defecto;
  - LGD: la red de lgd_entrenar.py (o la regresion lineal con --set perdida_esperada.modelo_lgd=lineal);
  - EAD: el monto desembolsado por la fraccion del capital que suele seguir pendiente cuando el
    credito se castiga (ver factor_exposicion).

La perdida observada de un credito es lo que se perdio de capital: EAD * LGD de lgd.parquet en los
castigados, y 0 en los que pagaron. En los "malos" que no estan castigados (Default y Late 31-120)
todavia no se sabe, asi que cuentan como 0 y la perdida observada queda como un piso.

Salidas en resultados/perdida_esperada/:
  - factor_exposicion.md, por_periodo.md, modelos.md, por_decil.md, ejemplos.md;
  - deciles.png;
  - creditos.parquet: PD, LGD, EAD y perdida esperada y observada de cada credito.

    python perdida_esperada.py
    python perdida_esperada.py --set perdida_esperada.ead=monto
"""

import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from riesgo import configuracion, datos, predecir_pd
from riesgo.red_numpy import predecir_numpy


def factor_exposicion(lgd, ultimo_anio):
    """Fraccion del monto desembolsado que seguia pendiente al castigo, por plazo.

    Se mide con suma(EAD) / suma(monto) y no con el promedio de cada credito, porque la perdida
    esperada se suma en dolares. Solo con los castigados de entrenamiento de la LGD.
    """
    usados = lgd[lgd["anio"] <= ultimo_anio]
    tabla = usados.groupby("plazo_meses").agg(castigados=("ead", "size"), ead=("ead", "sum"),
                                              monto=("funded_amnt", "sum"))
    tabla["factor"] = tabla["ead"] / tabla["monto"]
    return tabla


def predecir_lgd(df, modelo, carpeta="modelos"):
    """LGD predicha con el mismo preprocesamiento de lgd_entrenar.py, recortada a [0, 1]."""
    X = joblib.load(Path(carpeta) / "lgd_preprocesador.joblib").transform(df).astype(np.float32)
    if modelo == "red":
        return predecir_numpy(Path(carpeta) / "lgd_red.npz", X)
    # la regresion lineal no tiene salida acotada
    return np.clip(joblib.load(Path(carpeta) / "lgd_lineal.joblib").predict(X), 0, 1)


def creditos_con_perdida(cfg):
    """Todos los creditos con desenlace conocido, con su PD, LGD, EAD y perdidas."""
    df = datos.cargar(cfg["datos"]["procesados"], "pd")
    lgd = datos.cargar(cfg["datos"]["procesados"], "lgd")
    c = cfg["perdida_esperada"]

    factores = factor_exposicion(lgd, cfg["lgd"]["ultimo_anio_entrenamiento"])
    if c["ead"] == "factor":
        df["ead"] = df["funded_amnt"] * df["plazo_meses"].map(factores["factor"])
    else:
        df["ead"] = df["funded_amnt"].astype(float)

    modelos = predecir_pd.cargar("modelos")
    df["pd"] = predecir_pd.probabilidad(modelos, df)
    df["pd_logistica"] = predecir_pd.probabilidad(modelos, df, "logistica")
    df["lgd"] = predecir_lgd(df, c["modelo_lgd"])
    df["lgd_lineal"] = predecir_lgd(df, "lineal")
    df["perdida_esperada"] = df["pd"] * df["lgd"] * df["ead"]

    castigados = lgd.set_index("id")
    perdida = (castigados["ead"] * castigados["lgd"]).rename("perdida_observada")
    df = df.join(perdida, on="id")
    df["sin_perdida_conocida"] = (df["malo"] == 1) & df["perdida_observada"].isna()
    df["perdida_observada"] = df["perdida_observada"].fillna(0.0)
    return df, factores


def periodo(df, grupos):
    """Nombre del grupo de anios de cada credito."""
    nombres = pd.Series(pd.NA, index=df.index, dtype="object")
    for nombre, (desde, hasta) in grupos.items():
        nombres[df["anio"].between(desde, hasta)] = nombre
    return nombres


def porcentaje(x, decimales=1):
    return f"{100 * x:.{decimales}f}%"


def tabla_factor(factores):
    return pd.DataFrame({
        "Plazo": [f"{p:.0f} meses" for p in factores.index],
        "Castigados": [f"{n:,}".replace(",", " ") for n in factores["castigados"]],
        "Capital pendiente al castigo / monto": [porcentaje(f) for f in factores["factor"]],
    })


def tabla_por_periodo(df):
    """Perdida esperada contra observada, como % de lo desembolsado, en cada grupo de anios."""
    filas = []
    for nombre, d in df.groupby("periodo", sort=False):
        monto = d["funded_amnt"].sum()
        filas.append({
            "Originación": nombre,
            "Créditos": f"{len(d):,}".replace(",", " "),
            "Monto (millones de USD)": f"{monto / 1e6:,.0f}".replace(",", " "),
            "PD media": porcentaje(d["pd"].mean()),
            "Tasa de malos": porcentaje(d["malo"].mean()),
            "LGD media predicha": f"{d['lgd'].mean():.3f}",
            "EAD / monto": porcentaje(d["ead"].sum() / monto),
            "Pérdida esperada": porcentaje(d["perdida_esperada"].sum() / monto),
            "Pérdida esperada con EAD = monto": porcentaje((d["pd"] * d["lgd"] * d["funded_amnt"]).sum()
                                                           / monto),
            "Pérdida observada": porcentaje(d["perdida_observada"].sum() / monto),
            "Malos sin pérdida conocida": porcentaje(d["sin_perdida_conocida"].sum() / d["malo"].sum()),
        })
    return pd.DataFrame(filas)


def tabla_modelos(df, periodo_comparado):
    """Las cuatro combinaciones de modelos, en el periodo donde se puede comparar mejor."""
    d = df[df["periodo"] == periodo_comparado]
    monto = d["funded_amnt"].sum()
    observada = d["perdida_observada"].sum() / monto
    filas = []
    for nombre_pd, col_pd in [("Red", "pd"), ("Logística", "pd_logistica")]:
        for nombre_lgd, col_lgd in [("Red", "lgd"), ("Regresión lineal", "lgd_lineal")]:
            esperada = (d[col_pd] * d[col_lgd] * d["ead"]).sum() / monto
            filas.append({"PD": nombre_pd, "LGD": nombre_lgd,
                          "Pérdida esperada": porcentaje(esperada, 2),
                          "Pérdida observada": porcentaje(observada, 2),
                          "Esperada - observada (puntos %)": f"{100 * (esperada - observada):+.2f}"})
    return pd.DataFrame(filas)


def deciles(d):
    """Pérdida esperada y observada (% del monto) por decil de pérdida esperada por dólar."""
    tasa = d["perdida_esperada"] / d["funded_amnt"]
    decil = pd.qcut(tasa, 10, labels=range(1, 11))
    g = d.groupby(decil, observed=True)
    return pd.DataFrame({
        "esperada": g["perdida_esperada"].sum() / g["funded_amnt"].sum(),
        "observada": g["perdida_observada"].sum() / g["funded_amnt"].sum(),
        "malos": g["malo"].mean(),
    })


def tabla_por_decil(df, periodos):
    columnas = {}
    for p in periodos:
        t = deciles(df[df["periodo"] == p])
        columnas[f"Esperada ({p})"] = t["esperada"].map(porcentaje)
        columnas[f"Observada ({p})"] = t["observada"].map(porcentaje)
        columnas[f"Tasa de malos ({p})"] = t["malos"].map(porcentaje)
    tabla = pd.DataFrame(columnas)
    tabla.index.name = "Decil"
    return tabla.reset_index()


def figura_deciles(df, periodos, ruta):
    fig, ejes = plt.subplots(1, len(periodos), figsize=(5 * len(periodos), 4), sharey=True)
    for eje, p in zip(np.atleast_1d(ejes), periodos):
        t = deciles(df[df["periodo"] == p])
        x = np.arange(1, 11)
        eje.bar(x - 0.2, 100 * t["esperada"], 0.4, color="#2f5d8a", label="esperada")
        eje.bar(x + 0.2, 100 * t["observada"], 0.4, color="#e0a458", label="observada")
        eje.set_title(f"Originados en {p}", fontsize=10)
        eje.set_xlabel("decil de pérdida esperada (1 = menor)")
        eje.set_xticks(x)
    ejes[0].set_ylabel("pérdida (% del monto desembolsado)")
    ejes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def tabla_ejemplos(df, periodo_ejemplo):
    """Tres creditos de la prueba: pérdida esperada por dólar en los percentiles 10, 50 y 90."""
    d = df[df["periodo"] == periodo_ejemplo]
    tasa = d["perdida_esperada"] / d["funded_amnt"]
    filas = []
    for q in (0.10, 0.50, 0.90):
        fila = d.loc[(tasa - tasa.quantile(q)).abs().idxmin()]
        filas.append({
            "Percentil": f"{q:.0%}",
            "Monto": f"{fila['funded_amnt']:,.0f}".replace(",", " "),
            "Plazo": f"{fila['plazo_meses']:.0f} meses",
            "PD": porcentaje(fila["pd"]),
            "LGD": f"{fila['lgd']:.3f}",
            "EAD": f"{fila['ead']:,.0f}".replace(",", " "),
            "E[L] (USD)": f"{fila['perdida_esperada']:,.0f}".replace(",", " "),
            "E[L] / monto": porcentaje(fila["perdida_esperada"] / fila["funded_amnt"]),
            "Desenlace": "malo" if fila["malo"] == 1 else "bueno",
        })
    return pd.DataFrame(filas)


def main():
    a = configuracion.argumentos(__doc__).parse_args()
    cfg = configuracion.cargar(a.config, a.cambios)
    c = cfg["perdida_esperada"]
    salida = Path("resultados/demo/perdida_esperada" if a.demo else "resultados/perdida_esperada")
    salida.mkdir(parents=True, exist_ok=True)
    inicio = time.time()

    df, factores = creditos_con_perdida(cfg)
    if a.demo:
        df = df.sample(frac=cfg["datos"]["fraccion_demo"], random_state=cfg["semilla"])
    df["periodo"] = periodo(df, c["grupos"])
    periodos = list(c["grupos"])
    # 2013: la LGD es de prueba y casi todos los creditos ya terminaron; 2014-2015: prueba de la PD
    comparado, prueba = periodos[1], periodos[2]

    tablas = {
        "factor_exposicion.md": tabla_factor(factores),
        "por_periodo.md": tabla_por_periodo(df),
        "modelos.md": tabla_modelos(df, comparado),
        "por_decil.md": tabla_por_decil(df, [comparado, prueba]),
        "ejemplos.md": tabla_ejemplos(df, prueba),
    }
    for nombre, tabla in tablas.items():
        (salida / nombre).write_text(tabla.to_markdown(index=False, disable_numparse=True) + "\n",
                                     encoding="utf-8")
    figura_deciles(df, [comparado, prueba], salida / "deciles.png")
    columnas = ["id", "anio", "plazo_meses", "funded_amnt", "malo", "pd", "lgd", "ead",
                "perdida_esperada", "perdida_observada", "sin_perdida_conocida"]
    df[columnas].to_parquet(salida / "creditos.parquet", index=False)

    print(f"EAD: {c['ead']}; LGD: {c['modelo_lgd']}; PD: {cfg['scorecard']['modelo']}")
    for nombre in ("factor_exposicion.md", "por_periodo.md", "modelos.md"):
        print(tablas[nombre].to_markdown(index=False, disable_numparse=True))
    print(f"Tiempo total: {time.time() - inicio:.0f} s")


if __name__ == "__main__":
    main()
