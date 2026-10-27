"""Analisis descriptivo y prueba de las hipotesis sobre que hace riesgosa a una persona.

Las hipotesis se escribieron ANTES de mirar los datos (ver HIPOTESIS abajo).
Para cada una se compara la tasa de malos entre grupos y se dice si se confirma.

Salidas en resultados/descriptivo/: hipotesis.md, tasa_por_anio.png,
tasa_por_variable.png y lgd_distribucion.png.

    python descriptivo.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from riesgo import configuracion, datos

# (codigo, variable, como agrupar, que esperamos, sentido esperado)
#   sentido +1: a mayor valor (o en el grupo nombrado) mas riesgo; -1: menos riesgo
HIPOTESIS = [
    ("H1", "plazo_meses", "categoria", "Los creditos a 60 meses son mas riesgosos que a 36", +1),
    ("H2", "dti", "quintil", "Quien ya debe mas en proporcion a su ingreso es mas riesgoso", +1),
    ("H3", "annual_inc", "quintil", "A mayor ingreso, menor riesgo", -1),
    ("H4", "inq_last_6mths", "quintil", "Muchas consultas recientes de credito indican necesidad de plata y mas riesgo", +1),
    ("H5", "revol_util", "quintil", "Quien usa casi todo su cupo de tarjetas es mas riesgoso", +1),
    ("H6", "antiguedad_credito_meses", "quintil", "Una historia crediticia mas larga indica menos riesgo", -1),
    ("H7", "loan_amnt", "quintil", "Los prestamos mas grandes son mas riesgosos", +1),
    ("H8", "verification_status", "categoria", "Si el ingreso fue verificado, el riesgo es menor", -1),
    ("H9", "home_ownership", "categoria", "Quien arrienda es mas riesgoso que quien tiene casa o hipoteca", +1),
    ("H10", "antiguedad_empleo", "quintil", "Mas anios en el mismo empleo, menos riesgo", -1),
]

# para las categoricas: que grupo se compara contra cual (riesgoso esperado, otro)
COMPARACION = {
    "plazo_meses": (60.0, 36.0),
    "verification_status": ("Verified", "Not Verified"),
    "home_ownership": ("RENT", "MORTGAGE"),
}


def tasa_por_grupo(df, variable, agrupar):
    if agrupar == "quintil":
        grupos = pd.qcut(df[variable].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    else:
        grupos = df[variable]
    return df.groupby(grupos, observed=True)["malo"].agg(["mean", "size"])


def evaluar(df, codigo, variable, agrupar, texto, sentido):
    tabla = tasa_por_grupo(df, variable, agrupar)
    if agrupar == "quintil":
        alto, bajo = tabla["mean"].iloc[-1], tabla["mean"].iloc[0]
        detalle = f"quintil 5: {alto:.1%} vs quintil 1: {bajo:.1%}"
        diferencia = alto - bajo
    else:
        riesgoso, otro = COMPARACION[variable]
        alto, bajo = tabla.loc[riesgoso, "mean"], tabla.loc[otro, "mean"]
        detalle = f"{riesgoso}: {alto:.1%} vs {otro}: {bajo:.1%}"
        diferencia = (alto - bajo) if sentido > 0 else (bajo - alto)
        sentido = +1
    confirmada = diferencia * sentido > 0.01      # al menos 1 punto porcentual de diferencia
    return {"Hipótesis": codigo, "Qué esperábamos": texto, "Tasa de malos": detalle,
            "¿Se confirma?": "Sí" if confirmada else "No"}, tabla


def figura_por_anio(df, salida):
    por_anio = df.groupby("anio")["malo"].agg(["mean", "size"])
    fig, eje = plt.subplots(figsize=(7, 4))
    eje.bar(por_anio.index, por_anio["size"], color="#c9c3b6")
    eje.set_ylabel("créditos con desenlace conocido")
    eje2 = eje.twinx()
    eje2.plot(por_anio.index, 100 * por_anio["mean"], "o-", color="#b4532a")
    eje2.set_ylabel("tasa de malos (%)", color="#b4532a")
    eje.set_xlabel("año de originación")
    fig.tight_layout()
    fig.savefig(salida / "tasa_por_anio.png", dpi=130)
    plt.close(fig)


def figura_por_variable(tablas, salida):
    fig, ejes = plt.subplots(2, 5, figsize=(16, 6), sharey=True)
    for eje, ((codigo, variable, *_), tabla) in zip(ejes.flat, tablas):
        eje.bar([str(i) for i in tabla.index], 100 * tabla["mean"], color="#b4532a")
        eje.set_title(f"{codigo}: {variable}", fontsize=9)
        eje.tick_params(axis="x", labelsize=7, rotation=45)
    for eje in ejes[:, 0]:
        eje.set_ylabel("tasa de malos (%)")
    fig.tight_layout()
    fig.savefig(salida / "tasa_por_variable.png", dpi=130)
    plt.close(fig)


def figura_lgd(lgd, salida):
    fig, eje = plt.subplots(figsize=(7, 4))
    eje.hist(lgd["lgd"], bins=50, color="#2f6f4f")
    eje.set_xlabel("LGD (fracción de la exposición que se pierde)")
    eje.set_ylabel("créditos castigados")
    fig.tight_layout()
    fig.savefig(salida / "lgd_distribucion.png", dpi=130)
    plt.close(fig)


def main():
    a = configuracion.argumentos(__doc__).parse_args()
    cfg = configuracion.cargar(a.config, a.cambios)
    df = datos.cargar(cfg["datos"]["procesados"], "pd")
    lgd = datos.cargar(cfg["datos"]["procesados"], "lgd")
    salida = Path("resultados/descriptivo")
    salida.mkdir(parents=True, exist_ok=True)

    filas, tablas = [], []
    for h in HIPOTESIS:
        fila, tabla = evaluar(df, *h)
        filas.append(fila)
        tablas.append((h, tabla))

    figura_por_anio(df, salida)
    figura_por_variable(tablas, salida)
    figura_lgd(lgd, salida)

    md = pd.DataFrame(filas).to_markdown(index=False)
    (salida / "hipotesis.md").write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
