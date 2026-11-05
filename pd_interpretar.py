"""Que variables hacen mas riesgosa a una persona, con los modelos guardados (sin PyTorch).

Tres miradas que se complementan:
  1. Importancia por permutacion (Breiman, 2001; Fisher et al., 2019): se revuelve una variable
     entre los creditos y se mide cuanto cae el AUC. Si no cae, el modelo no la necesitaba.
     Sirve igual para la logistica y la red, y mide sobre datos que el modelo no vio. Dice
     cuanto importa una variable, pero no en que direccion.
  2. Coeficientes de la logistica: su signo da la direccion. En puntos, -factor * coeficiente.
  3. Dependencia parcial (Friedman, 2001): se le pone a TODOS los creditos el mismo valor de la
     variable y se promedia la PD. La diferencia entre dos valores da direccion y tamano en
     puntos porcentuales, con las demas variables como estan. Con esto se revisan H1 a H10.

Salidas en resultados/pd/: importancia.md, importancia.csv, importancia.png, importancia_grupos.md,
coeficientes.md, hipotesis_modelo.md, verificacion_anio_plazo.md, cliente_ejemplo.md y
correlaciones.md.

    python pd_interpretar.py
    python pd_interpretar.py --demo     # muestra del 10 % y 2 repeticiones (< 2 min)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from descriptivo import COMPARACION, HIPOTESIS, evaluar
from pd_figuras import CLAVES, COLORES, GRIS, TINTA_SUAVE, ejes_limpios, partes_evaluacion
from riesgo import configuracion, datos, metricas, predecir_pd, scorecard

# Una diferencia menor a esto (1 punto porcentual) no se cuenta como efecto: el mismo umbral
# con el que descriptivo.py decide si una hipotesis se confirma.
UMBRAL = 0.01

# Variables que dicen casi lo mismo. Al revolver solo una, la otra la cubre y las dos parecen
# menos importantes de lo que son: por eso tambien se revuelven juntas.
GRUPOS_RELACIONADOS = {
    "Cuentas: open_acc y total_acc": ["open_acc", "total_acc"],
    "Mora: delinq_2yrs, nunca_en_mora y meses_desde_mora": ["delinq_2yrs", "nunca_en_mora", "meses_desde_mora"],
    "Registros: pub_rec, sin_registros_publicos y meses_desde_registro": [
        "pub_rec", "sin_registros_publicos", "meses_desde_registro"],
}


# Una persona inventada, solo para mostrar como se reparten los puntos en el reporte.
CLIENTE_EJEMPLO = {
    "loan_amnt": 10000, "plazo_meses": 36, "antiguedad_empleo": 5, "annual_inc": 60000, "dti": 15,
    "delinq_2yrs": 0, "antiguedad_credito_meses": 180, "inq_last_6mths": 1, "meses_desde_mora": None,
    "meses_desde_registro": None, "open_acc": 10, "pub_rec": 0, "revol_bal": 10000, "revol_util": 50,
    "total_acc": 20, "collections_12_mths_ex_med": 0, "acc_now_delinq": 0, "home_ownership": "RENT",
    "verification_status": "Not Verified", "purpose": "debt_consolidation", "addr_state": "CA",
    "initial_list_status": "f"}


def tabla_cliente_ejemplo(modelos):
    """Los puntos de una misma persona con la red y con la logistica, y como suman el puntaje."""
    par = modelos["scorecard"]
    x = predecir_pd.transformar(modelos, predecir_pd.fila_cliente(modelos, CLIENTE_EJEMPLO))[0]
    referencia = modelos["referencia"]
    def en_puntaje(X, clave):
        return float(scorecard.puntaje(predecir_pd.predecir_matriz(modelos, X, clave)[0], par))
    red = scorecard.puntos_permutacion(lambda X: predecir_pd.predecir_matriz(modelos, X, "red"), x, referencia,
                                       modelos["grupos"], par, par["permutaciones"], modelos["semilla"])
    logistica = scorecard.puntos_logistica(modelos["logistica"], x, referencia, modelos["grupos"], par)
    filas = [{"Concepto": "Puntaje del cliente de referencia", "Red": f"{en_puntaje(referencia[None, :], 'red'):.1f}",
              "Logística": f"{en_puntaje(referencia[None, :], 'logistica'):.1f}"}]
    for v in sorted(red, key=lambda v: -abs(red[v])):
        filas.append({"Concepto": v, "Red": f"{red[v]:+.1f}", "Logística": f"{logistica[v]:+.1f}"})
    filas.append({"Concepto": "Puntaje de la persona", "Red": f"{en_puntaje(x[None, :], 'red'):.1f}",
                  "Logística": f"{en_puntaje(x[None, :], 'logistica'):.1f}"})
    filas.append({"Concepto": "PD de la persona", "Red": f"{predecir_pd.predecir_matriz(modelos, x[None, :], 'red')[0]:.1%}",
                  "Logística": f"{predecir_pd.predecir_matriz(modelos, x[None, :], 'logistica')[0]:.1%}"})
    return pd.DataFrame(filas)


def importancia_permutacion(modelos, df, repeticiones, semilla, grupos=None):
    """Caida media del AUC al revolver cada variable, para los dos modelos.

    Se revuelven todas las columnas de una variable a la vez (el one-hot completo), porque lo
    que se quiere saber es cuanto aporta la variable, no cada categoria por separado.
    """
    grupos = grupos or modelos["grupos"]
    X = predecir_pd.transformar(modelos, df)
    y = df["malo"].to_numpy()
    generador = np.random.default_rng(semilla)
    filas = {variable: {"Variable": variable} for variable in grupos}
    for nombre, clave in CLAVES.items():
        auc_base = metricas.auc(y, predecir_pd.predecir_matriz(modelos, X, clave))
        for variable, columnas in grupos.items():
            caidas = []
            for _ in range(repeticiones):
                revuelta = X.copy()
                revuelta[:, columnas] = X[generador.permutation(len(X))][:, columnas]
                caidas.append(auc_base - metricas.auc(y, predecir_pd.predecir_matriz(modelos, revuelta, clave)))
            filas[variable][nombre] = float(np.mean(caidas))
            filas[variable][f"{nombre} (desv)"] = float(np.std(caidas, ddof=1))
    return pd.DataFrame(filas.values())


def desviaciones(preprocesador):
    """Cuanto vale una desviacion estandar de cada numerica, en sus unidades."""
    salida = {}
    for nombre, pipeline, variables in preprocesador.transformers_:
        if nombre in ("log", "num"):
            sufijo = " en log(1+x)" if nombre == "log" else ""
            for v, escala in zip(variables, pipeline[-1].scale_):
                salida[v] = f"{escala:.3g}{sufijo}"
    return salida


def tabla_coeficientes(modelos):
    """Coeficientes de la logistica en orden de tamano, con los puntos que valen en la scorecard.

    Numericas: por cada desviacion estandar de mas. Categoricas: por tener esa categoria.
    El puntaje de la persona compara contra el promedio, asi que lo que cuenta en una variable
    categorica es la diferencia entre los puntos de sus categorias.
    """
    logistica = modelos["logistica"]
    f = scorecard.factor(modelos["scorecard"])
    escala = desviaciones(modelos["preprocesador"])
    filas = []
    for nombre, b in zip(modelos["preprocesador"].get_feature_names_out(), logistica.coef_[0]):
        columna = nombre.split("__", 1)[1]
        filas.append({"Columna": columna, "Una desviación equivale a": escala.get(columna, "-"),
                      "Coeficiente": b, "Puntos": -f * b})
    t = pd.DataFrame(filas)
    return t.reindex(t["Coeficiente"].abs().sort_values(ascending=False).index)


def diferencia_parcial(modelos, df, variable, alto, bajo, clave):
    """PD media si todos tuvieran 'alto' menos PD media si todos tuvieran 'bajo'."""
    con_alto = predecir_pd.probabilidad(modelos, df.assign(**{variable: alto}), clave).mean()
    con_bajo = predecir_pd.probabilidad(modelos, df.assign(**{variable: bajo}), clave).mean()
    return float(con_alto - con_bajo)


def tabla_hipotesis(modelos, df_todo, entrenamiento, validacion):
    """H1 a H10: lo que dicen los datos solos contra lo que dicen los modelos.

    En las numericas se comparan los percentiles 90 y 10 del entrenamiento (la mitad de los
    quintiles 5 y 1), en valores concretos de la variable.
    """
    filas = []
    for h in HIPOTESIS:
        codigo, variable, agrupar, texto, sentido = h
        descriptivo, _ = evaluar(df_todo, *h)
        if agrupar == "quintil":
            bajo, alto = np.nanquantile(entrenamiento[variable], [0.1, 0.9])
            comparacion = f"{alto:,.0f} vs {bajo:,.0f}".replace(",", " ")
        else:
            alto, bajo = COMPARACION[variable]
            comparacion = (f"{alto:.0f} vs {bajo:.0f} meses" if isinstance(alto, float)
                           else f"{alto} vs {bajo}")
        red = diferencia_parcial(modelos, validacion, variable, alto, bajo, "red")
        logistica = diferencia_parcial(modelos, validacion, variable, alto, bajo, "logistica")
        filas.append({"Hipótesis": codigo, "Variable": variable, "Sentido esperado": "sube" if sentido > 0 else "baja",
                      "Comparación": comparacion,
                      "Datos solos": descriptivo["¿Se confirma?"],
                      "Logística (puntos %)": f"{100 * logistica:+.1f}",
                      "Red (puntos %)": f"{100 * red:+.1f}",
                      "¿La red la confirma?": "Sí" if red * sentido > UMBRAL else "No"})
    return pd.DataFrame(filas)


def tabla_verificacion_anio_plazo(df):
    """H8 comparando solo creditos del mismo anio y plazo.

    La verificacion se volvio mas comun con los anios, y los anios recientes tienen mas malos por
    el sesgo de madurez. Comparar todo junto mezcla esos efectos con el de verificar.
    """
    d = df[df["verification_status"].isin(["Verified", "Not Verified"])]
    g = d.groupby(["anio", "plazo_meses", "verification_status"])["malo"].agg(["mean", "size"]).unstack().dropna()
    brecha = g[("mean", "Verified")] - g[("mean", "Not Verified")]
    peso = g[("size", "Verified")] + g[("size", "Not Verified")]
    tabla = pd.DataFrame({"Año": g.index.get_level_values(0),
                          "Plazo (meses)": g.index.get_level_values(1).astype(int),
                          "Créditos": peso.astype(int).to_numpy(),
                          "Malos: Verified": g[("mean", "Verified")].map("{:.1%}".format).to_numpy(),
                          "Malos: Not Verified": g[("mean", "Not Verified")].map("{:.1%}".format).to_numpy(),
                          "Brecha (puntos %)": (100 * brecha).map("{:+.1f}".format).to_numpy()})
    bruta = d[d["verification_status"] == "Verified"]["malo"].mean() - d[d["verification_status"] == "Not Verified"]["malo"].mean()
    ponderada = (brecha * peso).sum() / peso.sum()
    return tabla, f"Brecha bruta: {100 * bruta:+.1f} puntos. Dentro de cada anio y plazo, ponderada: {100 * ponderada:+.1f} puntos."


def tabla_correlaciones(df, variables):
    """Pares de variables numericas mas correlacionados (Spearman)."""
    c = df[variables].corr(method="spearman")
    pares = c.where(np.triu(np.ones(c.shape), 1).astype(bool)).stack()
    pares = pares.reindex(pares.abs().sort_values(ascending=False).index).head(5)
    return pd.DataFrame({"Variable 1": pares.index.get_level_values(0), "Variable 2": pares.index.get_level_values(1),
                         "Correlación de Spearman": pares.map("{:.2f}".format).to_numpy()})


def figura_importancia(tabla, ruta):
    """Barras horizontales ordenadas por la importancia en la red."""
    t = tabla.sort_values("Red neuronal")
    posiciones = np.arange(len(t))
    alto = 0.38
    fig, eje = plt.subplots(figsize=(8, 0.34 * len(t) + 1.2))
    for desplazamiento, nombre in [(alto / 2, "Red neuronal"), (-alto / 2, "Regresión logística")]:
        eje.barh(posiciones + desplazamiento, 100 * t[nombre], height=alto * 0.9,
                 color=COLORES[nombre], label=nombre)
    eje.set_yticks(posiciones, t["Variable"], fontsize=9)
    eje.set_xlabel("Caída del AUC al revolver la variable (puntos de AUC x 100)")
    eje.axvline(0, color=GRIS, linewidth=1)
    eje.legend(frameon=False, loc="lower right", fontsize=9)
    ejes_limpios(eje)
    eje.grid(axis="y", visible=False)
    eje.tick_params(axis="y", colors=TINTA_SUAVE, length=0)
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def formato_importancia(tabla):
    """La tabla del reporte: caida del AUC en validacion y en prueba, ordenada por la red."""
    t = tabla.sort_values("Red neuronal, prueba", ascending=False)
    return t.assign(**{c: t[c].map("{:.4f}".format) for c in t.columns if c != "Variable"})


def main():
    a = configuracion.argumentos(__doc__).parse_args()
    cfg = configuracion.cargar(a.config, a.cambios)
    repeticiones = 2 if a.demo else cfg["pd"]["repeticiones_permutacion"]
    salida = Path("resultados/demo/pd" if a.demo else "resultados/pd")
    salida.mkdir(parents=True, exist_ok=True)

    modelos = predecir_pd.cargar("modelos")
    entrenamiento, validacion, prueba = partes_evaluacion(cfg, a.demo)

    columnas = {}
    for particion, df in [("validación", validacion), ("prueba", prueba)]:
        t = importancia_permutacion(modelos, df, repeticiones, cfg["semilla"])
        columnas[particion] = t
        print(f"AUC cae en {particion}:")
        print(t.sort_values("Red neuronal", ascending=False).head(8).to_string(index=False))

    tabla = columnas["prueba"][["Variable"]].copy()
    for particion, t in columnas.items():
        for nombre in CLAVES:
            tabla[f"{nombre}, {particion}"] = t[nombre]
    tabla.to_csv(salida / "importancia.csv", index=False)
    (salida / "importancia.md").write_text(formato_importancia(tabla).to_markdown(index=False, disable_numparse=True) + "\n",
                                           encoding="utf-8")
    figura_importancia(columnas["prueba"], salida / "importancia.png")

    grupos = {nombre: sum((modelos["grupos"][v] for v in variables), [])
              for nombre, variables in GRUPOS_RELACIONADOS.items()}
    por_grupo = pd.DataFrame({"Variables revueltas juntas": list(grupos)})
    for particion, df in [("validación", validacion), ("prueba", prueba)]:
        t = importancia_permutacion(modelos, df, repeticiones, cfg["semilla"], grupos)
        for nombre in CLAVES:
            por_grupo[f"{nombre}, {particion}"] = t[nombre].map("{:.4f}".format)
    (salida / "importancia_grupos.md").write_text(por_grupo.to_markdown(index=False, disable_numparse=True) + "\n", encoding="utf-8")
    print(por_grupo.to_string(index=False))

    rho, _ = spearmanr(columnas["prueba"]["Red neuronal"], columnas["prueba"]["Regresión logística"])
    rho_val, _ = spearmanr(columnas["validación"]["Red neuronal"], columnas["prueba"]["Red neuronal"])
    print(f"Correlacion de rangos red vs logistica (prueba): {rho:.3f}")
    print(f"Correlacion de rangos validacion vs prueba (red): {rho_val:.3f}")

    coeficientes = tabla_coeficientes(modelos)
    coef_md = coeficientes.assign(Coeficiente=coeficientes["Coeficiente"].map("{:+.4f}".format),
                                  Puntos=coeficientes["Puntos"].map("{:+.1f}".format))
    (salida / "coeficientes.md").write_text(coef_md.to_markdown(index=False, disable_numparse=True) + "\n", encoding="utf-8")
    print(coef_md.head(15).to_string(index=False))

    df_todo = datos.cargar(cfg["datos"]["procesados"], "pd")
    hipotesis = tabla_hipotesis(modelos, df_todo, entrenamiento, validacion)
    (salida / "hipotesis_modelo.md").write_text(hipotesis.to_markdown(index=False, disable_numparse=True) + "\n", encoding="utf-8")
    print(hipotesis.to_string(index=False))

    verificacion, resumen = tabla_verificacion_anio_plazo(df_todo)
    (salida / "verificacion_anio_plazo.md").write_text(
        verificacion.to_markdown(index=False, disable_numparse=True) + f"\n\n{resumen}\n", encoding="utf-8")
    print(resumen)

    ejemplo = tabla_cliente_ejemplo(modelos)
    (salida / "cliente_ejemplo.md").write_text(ejemplo.to_markdown(index=False, disable_numparse=True) + "\n",
                                               encoding="utf-8")
    print(ejemplo.to_string(index=False))

    correlaciones = tabla_correlaciones(df_todo, cfg["variables"]["numericas"])
    (salida / "correlaciones.md").write_text(correlaciones.to_markdown(index=False, disable_numparse=True) + "\n", encoding="utf-8")
    print(correlaciones.to_string(index=False))


if __name__ == "__main__":
    main()
