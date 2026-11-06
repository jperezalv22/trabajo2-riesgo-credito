"""Aplicacion web: puntaje de credito, percentil y "que pasaria si...".

    streamlit run app/app.py

Solo usa la interfaz de riesgo/predecir_pd.py (cargar y evaluar_cliente). No importa torch:
la red se evalua con NumPy (riesgo/red_numpy.py).
"""

import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
# streamlit solo agrega app/ a la ruta de importacion, y el paquete riesgo/ esta en la raiz
sys.path.insert(0, str(RAIZ))

import etiquetas as et  # noqa: E402
from riesgo import configuracion, predecir_pd  # noqa: E402

DINERO = ["annual_inc", "loan_amnt", "revol_bal"]
PORCENTAJE = ["dti", "revol_util"]
SI_NO = ["nunca_en_mora", "sin_registros_publicos"]
# si la persona nunca ha tenido mora (o registros), la variable de meses queda vacia, como en
# los datos: el modelo la imputa con la mediana y el indicador dice que nunca paso (HANDOFF D4)
DEPENDIENTES = {"meses_desde_mora": "nunca_en_mora",
                "meses_desde_registro": "sin_registros_publicos"}


# ---------- carga (una sola vez por servidor) ----------

@st.cache_resource
def cargar_modelos():
    return predecir_pd.cargar(RAIZ / "modelos")


@st.cache_resource
def cargar_config():
    return configuracion.cargar(RAIZ / "config.yaml")


@st.cache_data
def valores_tipicos(numericas, categoricas, casos_minimos):
    """Mediana de cada numerica y categorias ordenadas de la mas comun a la menos comun."""
    df = pd.read_parquet(RAIZ / "datos/procesados/pd.parquet", columns=numericas + categoricas)
    tipicos = {v: float(df[v].median()) for v in numericas}
    opciones = {}
    for v in categoricas:
        conteo = df[v].value_counts()
        opciones[v] = conteo[conteo >= casos_minimos].index.tolist()
    return tipicos, opciones


@st.cache_data
def histograma_poblacion(_puntajes, barras):
    conteos, bordes = np.histogram(_puntajes, bins=barras)
    return pd.DataFrame({"desde": bordes[:-1], "hasta": bordes[1:], "personas": conteos})


# ---------- formulario ----------

def texto_valor(variable, valor):
    """Valor de una variable como lo leeria una persona."""
    if valor is None:
        return "no aplica"
    if variable in SI_NO:
        return "No" if valor == 1 else "Sí"
    if variable in DINERO:
        return f"{valor:,.0f} dólares"
    if variable in PORCENTAJE:
        return f"{valor:.1f} %"
    if variable == "antiguedad_credito_meses":
        return f"{valor / 12:.0f} años"
    if variable == "antiguedad_empleo":
        return {0: "menos de 1 año", 10: "10 años o más"}.get(valor, f"{valor:.0f} años")
    if isinstance(valor, str):
        return et.categoria(variable, valor)
    return f"{valor:.0f}"


def entrada(variable, valor, clave, opciones):
    """Dibuja la entrada que corresponde a la variable y devuelve el valor elegido."""
    texto, ayuda = et.pregunta(variable), et.ayuda(variable)
    if variable in opciones:
        lista = opciones[variable]
        return st.selectbox(texto, lista, index=lista.index(valor), key=clave, help=ayuda,
                            format_func=lambda c: et.categoria(variable, c))
    if variable in SI_NO:
        # 1 en el indicador = nunca ha pasado = respuesta "No"
        respuesta = st.radio(texto, ["No", "Sí"], index=0 if valor == 1 else 1, key=clave,
                             help=ayuda, horizontal=True)
        return 1 if respuesta == "No" else 0
    if variable == "plazo_meses":
        return st.radio(texto, [36, 60], index=[36, 60].index(int(valor)), key=clave,
                        help=ayuda, horizontal=True)
    if variable == "antiguedad_empleo":
        return st.select_slider(texto, list(range(11)), value=int(valor), key=clave, help=ayuda,
                                format_func=lambda a: texto_valor(variable, a))
    if variable == "antiguedad_credito_meses":
        # se pregunta en anios, que es como la gente lo sabe; el modelo usa meses
        anios = st.number_input(texto, 0, 80, int(round(valor / 12)), key=clave, help=ayuda)
        return anios * 12.0
    if variable in DINERO:
        return float(st.number_input(texto, 0, None, int(valor), step=500, key=clave, help=ayuda))
    if variable in PORCENTAJE:
        return st.number_input(texto, 0.0, 200.0, float(valor), step=1.0, key=clave, help=ayuda)
    return float(st.number_input(texto, 0, None, int(valor), step=1, key=clave, help=ayuda))


def valor_inicial(variable, tipicos, opciones):
    if variable in opciones:
        return opciones[variable][0]
    return tipicos[variable]


def formulario(tipicos, opciones):
    """Las preguntas en tres pestanas; devuelve el cliente como diccionario."""
    cliente = {}
    pestanas = st.tabs(list(et.GRUPOS))
    for pestana, variables in zip(pestanas, et.GRUPOS.values()):
        with pestana:
            for v in variables:
                if v in DEPENDIENTES and cliente[DEPENDIENTES[v]] == 1:
                    cliente[v] = None
                    continue
                cliente[v] = entrada(v, valor_inicial(v, tipicos, opciones), f"f_{v}", opciones)
    return cliente


# ---------- resultados ----------

def mostrar_resultado(r):
    a, b, c = st.columns(3)
    a.metric("Su puntaje", f"{r['puntaje']:.0f}")
    b.metric("Probabilidad de no pagar", f"{100 * r['pd']:.1f} %")
    c.metric("Su percentil", f"{r['percentil']:.0f}")
    st.write(f"De cada 100 personas con un perfil como el suyo, el modelo espera que cerca de "
             f"**{100 * r['pd']:.0f}** no paguen su crédito. Su puntaje es igual o mayor que el de "
             f"**{r['percentil']:.0f} de cada 100** personas de la población. "
             f"Más puntaje es menos riesgo.")


def grafico_poblacion(tabla, puntaje):
    tabla = tabla.assign(grupo=np.where(tabla["hasta"] <= puntaje, "Puntaje menor o igual al suyo",
                                        "Puntaje mayor al suyo"))
    barras = alt.Chart(tabla).mark_bar().encode(
        x=alt.X("desde:Q", title="Puntaje", bin="binned"), x2="hasta:Q",
        y=alt.Y("personas:Q", title="Personas"),
        color=alt.Color("grupo:N", title=None, legend=alt.Legend(orient="bottom"),
                        scale=alt.Scale(range=["#9aa7b4", "#1f6feb"])))
    linea = alt.Chart(pd.DataFrame({"x": [puntaje]})).mark_rule(
        color="#d1242f", strokeWidth=3).encode(x="x:Q")
    texto = alt.Chart(pd.DataFrame({"x": [puntaje], "t": ["Usted"]})).mark_text(
        dy=-8, color="#d1242f", fontWeight="bold", align="left", dx=4).encode(
        x="x:Q", y=alt.value(10), text="t:N")
    st.altair_chart(barras + linea + texto, width="stretch")


def grafico_puntos(r, cliente, cuantos=10):
    """Las variables que mas suben o bajan el puntaje frente al cliente promedio."""
    filas = [{"Característica": f"{et.pregunta(v)}: {texto_valor(v, cliente.get(v))}",
              "Puntos": p} for v, p in r["puntos"].items()]
    tabla = pd.DataFrame(filas)
    tabla = tabla.reindex(tabla["Puntos"].abs().sort_values(ascending=False).index).head(cuantos)
    tabla["Efecto"] = np.where(tabla["Puntos"] >= 0, "Le suma", "Le resta")
    grafico = alt.Chart(tabla).mark_bar().encode(
        x=alt.X("Puntos:Q"), y=alt.Y("Característica:N", sort=None, title=None),
        color=alt.Color("Efecto:N", title=None, legend=alt.Legend(orient="bottom"),
                        scale=alt.Scale(domain=["Le suma", "Le resta"],
                                        range=["#1a7f37", "#d1242f"])),
        tooltip=["Característica", alt.Tooltip("Puntos:Q", format="+.1f")])
    st.altair_chart(grafico, width="stretch")
    st.caption(f"Puntos frente a una persona promedio, que tiene {r['puntos_base']:.0f}. "
               f"Esos {r['puntos_base']:.0f} más los puntos de todas sus características dan su "
               f"puntaje ({r['puntaje']:.0f}).")


def que_pasaria(modelos, cliente, r, tipicos, opciones, no_sugerir):
    """La persona cambia una caracteristica y ve el puntaje nuevo y la diferencia."""
    posibles = [v for vs in et.GRUPOS.values() for v in vs
                if v not in no_sugerir and cliente.get(v) is not None]
    variable = st.selectbox("¿Qué característica quiere cambiar?", posibles,
                            format_func=et.pregunta)
    nuevo_valor = entrada(variable, cliente[variable], f"q_{variable}", opciones)
    otro = dict(cliente, **{variable: nuevo_valor})
    # al pasar a "Sí ha estado en mora", hay que decir hace cuantos meses
    for meses, indicador in DEPENDIENTES.items():
        if variable == indicador and nuevo_valor == 0 and otro[meses] is None:
            otro[meses] = entrada(meses, tipicos[meses], f"q_{meses}", opciones)
        if variable == indicador and nuevo_valor == 1:
            otro[meses] = None
    nuevo = predecir_pd.evaluar_cliente(modelos, otro)
    a, b = st.columns(2)
    a.metric("Puntaje nuevo", f"{nuevo['puntaje']:.0f}",
             f"{nuevo['puntaje'] - r['puntaje']:+.1f} puntos")
    b.metric("Probabilidad de no pagar", f"{100 * nuevo['pd']:.1f} %",
             f"{100 * (nuevo['pd'] - r['pd']):+.1f} puntos porcentuales", delta_color="inverse")
    st.caption(f"Con {texto_valor(variable, cliente[variable])} su puntaje es "
               f"{r['puntaje']:.0f}; con {texto_valor(variable, nuevo_valor)} sería "
               f"{nuevo['puntaje']:.0f}. Es lo que dice el modelo con todo lo demás igual.")


def listas_variables():
    v = cargar_config()["variables"]
    return list(v["numericas"]), list(v["categoricas"])


def enlaces(cfg):
    e = cfg["app"]["enlaces"]
    video = f"[Video promocional]({e['video']})" if e["video"] else "Video promocional (pronto)"
    st.markdown(f"[Reporte técnico (blog)]({e['blog']}) · {video} · "
                f"[Código en GitHub]({e['repositorio']})")


# ---------- pagina ----------

def main():
    st.set_page_config(page_title="¿Me prestas? Su puntaje de crédito", page_icon="💳",
                       layout="wide")
    cfg = cargar_config()
    modelos = cargar_modelos()
    tipicos, opciones = valores_tipicos(*listas_variables(), cfg["app"]["casos_minimos_categoria"])

    st.title("¿Me prestas? Conozca su puntaje de crédito")
    st.write("Antes de prestarle, un banco se pregunta qué tan probable es que usted no pague. "
             "Esta aplicación le muestra lo que respondería un modelo entrenado con 130 832 "
             "créditos de Lending Club, originados entre 2007 y 2013.")
    st.info("**Cómo se usa:** 1) cambie sus datos a la izquierda (ya vienen llenos con los de una "
            "persona típica); 2) mire su puntaje a la derecha; 3) abajo, pruebe qué pasaría si "
            "cambiara algo.")
    enlaces(cfg)

    izquierda, derecha = st.columns([1, 1.3], gap="large")
    with izquierda:
        st.subheader("1. Sus datos")
        cliente = formulario(tipicos, opciones)
    r = predecir_pd.evaluar_cliente(modelos, cliente)
    with derecha:
        st.subheader("2. Su resultado")
        mostrar_resultado(r)
        st.markdown("**Dónde queda frente a los demás**")
        grafico_poblacion(histograma_poblacion(modelos["puntajes"],
                                               cfg["app"]["barras_histograma"]), r["puntaje"])
        st.markdown("**Qué le suma y qué le resta puntos**")
        grafico_puntos(r, cliente)

    st.divider()
    st.subheader("3. ¿Qué pasaría si...?")
    que_pasaria(modelos, cliente, r, tipicos, opciones, cfg["app"]["no_sugerir"])
    st.caption("No se ofrece cambiar la verificación del ingreso: en los datos, los créditos "
               "verificados tienen más malos por cómo Lending Club escogía a quién verificar, no "
               "porque verificar aumente el riesgo (secciones 3 y 5 del reporte).")

    st.divider()
    st.caption("Ejercicio académico del curso Redes Neuronales y Algoritmos Bioinspirados, "
               "Universidad Nacional de Colombia, 2026-2. El modelo se entrenó con créditos de "
               "Lending Club (Estados Unidos) originados entre 2007 y 2013: no es una decisión de "
               "crédito real.")


main()
