"""Analisis descriptivo y prueba de las hipotesis sobre que hace riesgosa a una persona.

Las hipotesis se escribieron ANTES de mirar los datos (ver HIPOTESIS abajo).
Para cada una se compara la tasa de malos entre grupos y se dice si se confirma.

Salidas en resultados/descriptivo/:
  - hipotesis.md y verificacion.md (H8 por quintil de monto);
  - estados.md, por_anio.md, variables.md, particiones_pd.md: tablas de la seccion de datos;
  - lgd_por_anio.md, particiones_lgd.md, recuperaciones.md: tablas de la LGD;
  - tasa_por_anio.png, tasa_por_variable.png y lgd_distribucion.png.

    python descriptivo.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from riesgo import configuracion, datos, particiones, preparar

# (codigo, variable, como agrupar, que esperamos, sentido esperado)
#   sentido +1: a mayor valor (o en el grupo nombrado) mas riesgo; -1: menos riesgo
HIPOTESIS = [
    ("H1", "plazo_meses", "categoria", "Los créditos a 60 meses son más riesgosos que a 36", +1),
    ("H2", "dti", "quintil", "Quien ya debe más en proporción a su ingreso es más riesgoso", +1),
    ("H3", "annual_inc", "quintil", "A mayor ingreso, menor riesgo", -1),
    ("H4", "inq_last_6mths", "quintil", "Muchas consultas recientes de crédito indican necesidad de plata y más riesgo", +1),
    ("H5", "revol_util", "quintil", "Quien usa casi todo su cupo de tarjetas es más riesgoso", +1),
    ("H6", "antiguedad_credito_meses", "quintil", "Una historia crediticia más larga indica menos riesgo", -1),
    ("H7", "loan_amnt", "quintil", "Los préstamos más grandes son más riesgosos", +1),
    ("H8", "verification_status", "categoria", "Si el ingreso fue verificado, el riesgo es menor", -1),
    ("H9", "home_ownership", "categoria", "Quien arrienda es más riesgoso que quien tiene casa o hipoteca", +1),
    ("H10", "antiguedad_empleo", "quintil", "Más años en el mismo empleo, menos riesgo", -1),
]

# para las categoricas: que grupo se compara contra cual (riesgoso esperado, otro)
COMPARACION = {
    "plazo_meses": (60.0, 36.0),
    "verification_status": ("Verified", "Not Verified"),
    "home_ownership": ("RENT", "MORTGAGE"),
}


def quintiles(serie):
    """Quintil (1 a 5) de cada valor; rank() reparte los empates para que queden 5 grupos iguales."""
    return pd.qcut(serie.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])


def nombre_grupo(valor):
    """El plazo es un numero (60.0); se muestra como "60 meses"."""
    return f"{valor:.0f} meses" if isinstance(valor, float) else valor


def tasa_por_grupo(df, variable, agrupar):
    if agrupar == "quintil":
        grupos = quintiles(df[variable])
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
        detalle = f"{nombre_grupo(riesgoso)}: {alto:.1%} vs {nombre_grupo(otro)}: {bajo:.1%}"
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


def figura_lgd(lgd, ultimo_anio, salida):
    """LGD de los anios que se usan contra la de los recientes, que todavia no recuperan (D7)."""
    usados = lgd["anio"] <= ultimo_anio
    fig, eje = plt.subplots(figsize=(7, 4))
    eje.hist(lgd.loc[usados, "lgd"], bins=50, range=(0, 1), density=True, color="#2f6f4f",
             alpha=0.6, label=f"originados hasta {ultimo_anio} (se usan)")
    eje.hist(lgd.loc[~usados, "lgd"], bins=50, range=(0, 1), density=True, histtype="step",
             linewidth=2, color="#b4532a", label=f"originados después de {ultimo_anio} (se excluyen)")
    eje.set_xlabel("LGD (fracción de la exposición que se pierde)")
    eje.set_ylabel("densidad")
    eje.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(salida / "lgd_distribucion.png", dpi=130)
    plt.close(fig)


# --- Tablas para el reporte ---------------------------------------------------

# variable del modelo -> (columna del archivo original, como se transforma)
ORIGEN = {
    "plazo_meses": ("term", '"36 months" -> 36'),
    "antiguedad_empleo": ("emp_length", '"< 1 year" -> 0, "10+ years" -> 10'),
    "antiguedad_credito_meses": ("earliest_cr_line", "meses entre esa fecha y la originación"),
    "meses_desde_mora": ("mths_since_last_delinq", "vacío = nunca (ver nunca_en_mora)"),
    "meses_desde_registro": ("mths_since_last_record", "vacío = nunca (ver sin_registros_publicos)"),
    "nunca_en_mora": ("mths_since_last_delinq", "1 si está vacío, 0 si no"),
    "sin_registros_publicos": ("mths_since_last_record", "1 si está vacío, 0 si no"),
}
# en el diccionario, verification_status aparece con su nombre anterior
EN_DICCIONARIO = {"verification_status": "is_inc_v"}


def miles(n):
    return f"{int(n):,}".replace(",", " ")


def porcentaje(x):
    return f"{x:.1%}"


def tabla_estados(estados):
    """Cuantos creditos hay de cada loan_status y como se codifican (tabla del enunciado)."""
    total = estados.sum().sort_values(ascending=False)
    return pd.DataFrame({"loan_status": total.index,
                         "Codificación": [datos.MALO.get(e, "NA") for e in total.index],
                         "Créditos": total.map(miles).values})


def tabla_por_anio(estados, df, ultimo_anio):
    """Por anio: originados, cuantos tienen desenlace, tasa de malos y peso de los de 60 meses."""
    g = df.groupby("anio")
    conocidos = g.size()
    return pd.DataFrame({
        "Año": conocidos.index,
        "Partición": ["entrenamiento y validación" if a <= ultimo_anio else "prueba"
                      for a in conocidos.index],
        "Originados": estados.sum(axis=1).loc[conocidos.index].map(miles).values,
        "Con desenlace conocido": conocidos.map(miles).values,
        "% con desenlace": (conocidos / estados.sum(axis=1)).loc[conocidos.index].map(porcentaje).values,
        "Tasa de malos": g["malo"].mean().map(porcentaje).values,
        "A 60 meses": g["plazo_meses"].apply(lambda s: (s == 60).mean()).map(porcentaje).values,
    })


def tabla_variables(cfg, df):
    """Cada variable de entrada con su definicion textual del diccionario de Lending Club."""
    dic = pd.read_excel(cfg["datos"]["diccionario"], sheet_name="LoanStats").dropna()
    definicion = dict(zip(dic["LoanStatNew"].str.strip(), dic["Description"].str.strip()))
    filas = []
    for var in cfg["variables"]["numericas"] + cfg["variables"]["categoricas"]:
        original, cambio = ORIGEN.get(var, (var, ""))
        if var in preparar.CON_LOGARITMO:
            cambio = "log(1 + x)"
        if original in EN_DICCIONARIO:
            cambio = f"en el diccionario se llama {EN_DICCIONARIO[original]}"
        filas.append({"Variable": var, "Columna original": original,
                      "Definición en el diccionario": definicion[EN_DICCIONARIO.get(original, original)],
                      "Transformación": cambio, "Faltantes": porcentaje(df[var].isna().mean())})
    return pd.DataFrame(filas)


def tabla_particiones(df, ultimo_anio, fraccion, semilla, objetivo, nombre, formato,
                      estratificar=None):
    """Las mismas particiones que usan los modelos, con su tamano y el promedio del objetivo."""
    partes = particiones.dividir(df, ultimo_anio, fraccion, semilla, estratificar)
    t = pd.DataFrame(particiones.resumen(*partes, objetivo=objetivo))
    t.columns = ["Partición", "Créditos", "Años", nombre]
    t["Créditos"] = t["Créditos"].map(miles)
    t[nombre] = t[nombre].map(formato)
    return t


def uso_lgd(anio, c):
    if anio <= c["ultimo_anio_entrenamiento"]:
        return "entrenamiento y validación"
    if anio <= c["ultimo_anio_datos"]:
        return "prueba"
    return "se excluye: cobro sin terminar"


def tabla_lgd_por_anio(lgd, c):
    """LGD por anio de originacion y para que se usa cada anio (HANDOFF D7)."""
    g = lgd.groupby("anio")["lgd"]
    n = g.size()
    return pd.DataFrame({"Año": n.index, "Uso": [uso_lgd(a, c) for a in n.index],
                         "Castigados": n.map(miles).values,
                         "LGD media": g.mean().map("{:.3f}".format).values,
                         "LGD = 1": g.apply(lambda s: (s == 1).mean()).map(porcentaje).values})


def tabla_recuperaciones(lgd, ultimo_anio):
    """Valores mas frecuentes de lo recuperado sobre la EAD y de la comision de cobro."""
    l = lgd[lgd["anio"] <= ultimo_anio]
    bruto = (l["recoveries"] / l["ead"]).round(2).value_counts(normalize=True).head(5)
    con_algo = l[l["recoveries"] > 0]
    comision = (con_algo["collection_recovery_fee"] / con_algo["recoveries"]).round(2)
    comision = comision.value_counts(normalize=True).head(4)
    t1 = pd.DataFrame({"Recuperación bruta / EAD (2 decimales)": bruto.index,
                       "% de los castigados": bruto.map(porcentaje).values})
    t2 = pd.DataFrame({"Comisión de cobro / recuperación (2 decimales)": comision.index,
                       "% de los que recuperaron algo": comision.map(porcentaje).values})
    return t1.to_markdown(index=False) + "\n\n" + t2.to_markdown(index=False)


def tabla_verificacion(df):
    """H8 por quintil de monto: cuantos no se verifican y la tasa de malos de cada estado."""
    q = quintiles(df["loan_amnt"]).rename("Quintil de monto")
    sin_verificar = pd.crosstab(q, df["verification_status"], normalize="index")["Not Verified"]
    tasas = df.pivot_table(index=q, columns="verification_status", values="malo",
                           aggfunc="mean", observed=True).map(porcentaje)
    tasas.columns = [f"Malos: {c}" for c in tasas.columns]
    tasas.insert(0, "Monto máximo", df.groupby(q, observed=True)["loan_amnt"].max().map(miles))
    tasas.insert(1, "Sin verificar", sin_verificar.map(porcentaje))
    return tasas.reset_index()


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

    p, c = cfg["particion"], cfg["lgd"]
    figura_por_anio(df, salida)
    figura_por_variable(tablas, salida)
    figura_lgd(lgd, c["ultimo_anio_datos"], salida)

    estados = pd.read_csv(Path(cfg["datos"]["procesados"]) / "estados_por_anio.csv",
                          index_col="anio")
    lgd_usada = lgd[lgd["anio"] <= c["ultimo_anio_datos"]]
    tablas_md = {
        "hipotesis.md": pd.DataFrame(filas),
        "verificacion.md": tabla_verificacion(df),
        "estados.md": tabla_estados(estados),
        "por_anio.md": tabla_por_anio(estados, df, p["ultimo_anio_entrenamiento"]),
        "variables.md": tabla_variables(cfg, df),
        "particiones_pd.md": tabla_particiones(df, p["ultimo_anio_entrenamiento"],
                                               p["fraccion_validacion"], cfg["semilla"],
                                               "malo", "Tasa de malos", porcentaje,
                                               estratificar="malo"),
        "particiones_lgd.md": tabla_particiones(lgd_usada, c["ultimo_anio_entrenamiento"],
                                                c["fraccion_validacion"], cfg["semilla"],
                                                "lgd", "LGD media", "{:.3f}".format),
        "lgd_por_anio.md": tabla_lgd_por_anio(lgd, c),
    }
    for nombre, tabla in tablas_md.items():
        (salida / nombre).write_text(tabla.to_markdown(index=False) + "\n", encoding="utf-8")
    (salida / "recuperaciones.md").write_text(
        tabla_recuperaciones(lgd, c["ultimo_anio_datos"]) + "\n", encoding="utf-8")
    print(tablas_md["hipotesis.md"].to_markdown(index=False))


if __name__ == "__main__":
    main()
