"""Tres preguntas de sensibilidad sobre las decisiones de la PD, con la regresion logistica.

Se usa la logistica porque se reajusta en segundos y aqui lo que importa es la comparacion:
  1. Pesos por clase: ¿ayudan a detectar malos? ¿que le hacen a la calibracion?
  2. initial_list_status: casi no existia antes de 2012 (D3 saco variables por algo parecido).
     ¿Cuanto depende el modelo de ella?
  3. grade, sub_grade, int_rate e installment, excluidas en D3 porque salen del modelo de
     Lending Club: ¿cuanta senal deja por fuera esa decision?

Cada escenario elige su C en validacion, con la misma rejilla del modelo de referencia.
Salida: resultados/pd/sensibilidad.md y lista_por_anio.md. No modifica los modelos de modelos/.

    python pd_sensibilidad.py
    python pd_sensibilidad.py --demo    # muestra del 10 %
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from riesgo import configuracion, datos, metricas, particiones, preparar

ELEGIDAS_POR_LC_NUMERICAS = ["int_rate", "installment"]
ELEGIDAS_POR_LC_CATEGORICAS = ["grade", "sub_grade"]


def evaluar(partes, numericas, categoricas, valores_c, max_iter, pesos=None):
    """Ajusta la logistica con este escenario y mide en validacion y prueba."""
    prepro = preparar.crear_preprocesador(numericas, categoricas)
    m = preparar.matrices(prepro, *partes, objetivo="malo")
    X_ent, y_ent = m["entrenamiento"]
    X_val, y_val = m["validacion"]
    X_pru, y_pru = m["prueba"]
    mejor = None
    for c in valores_c:
        modelo = LogisticRegression(C=c, max_iter=max_iter, class_weight=pesos).fit(X_ent, y_ent)
        perdida = metricas.entropia_cruzada(y_val, modelo.predict_proba(X_val)[:, 1])
        if mejor is None or perdida < mejor[0]:
            mejor = (perdida, modelo)
    modelo = mejor[1]
    p_val, p_pru = modelo.predict_proba(X_val)[:, 1], modelo.predict_proba(X_pru)[:, 1]
    r_val, r_pru = metricas.clasificacion(y_val, p_val), metricas.clasificacion(y_pru, p_pru)
    return {"C": modelo.C, "Parámetros": modelo.coef_.size + 1,
            "AUC validación": f"{r_val['auc']:.4f}", "Brier validación": f"{r_val['brier']:.4f}",
            "PD media validación": f"{np.mean(p_val):.1%}",
            "AUC prueba": f"{r_pru['auc']:.4f}", "KS prueba": f"{r_pru['ks']:.4f}",
            "Brier prueba": f"{r_pru['brier']:.4f}", "PD media prueba": f"{np.mean(p_pru):.1%}"}


def main():
    a = configuracion.argumentos(__doc__).parse_args()
    cfg = configuracion.cargar(a.config, a.cambios)
    fraccion = cfg["datos"]["fraccion_demo"] if a.demo else 1.0
    df = datos.cargar(cfg["datos"]["procesados"], "pd", fraccion, cfg["semilla"])
    p = cfg["particion"]
    partes = particiones.dividir(df, p["ultimo_anio_entrenamiento"], p["fraccion_validacion"],
                                 cfg["semilla"], estratificar="malo")
    numericas, categoricas = cfg["variables"]["numericas"], cfg["variables"]["categoricas"]
    c = cfg["pd"]["logistica"]
    valores_c, max_iter = c["C"], c["max_iteraciones"]

    sin_lista = [v for v in categoricas if v != "initial_list_status"]
    escenarios = [
        ("Modelo de referencia", numericas, categoricas, None),
        ("Con pesos por clase (balanced)", numericas, categoricas, "balanced"),
        ("Sin initial_list_status", numericas, sin_lista, None),
        ("Con grade, sub_grade, int_rate e installment", numericas + ELEGIDAS_POR_LC_NUMERICAS,
         categoricas + ELEGIDAS_POR_LC_CATEGORICAS, None),
    ]
    filas = []
    for nombre, num, cat, pesos in escenarios:
        filas.append({"Escenario": nombre, **evaluar(partes, num, cat, valores_c, max_iter, pesos)})
        print(filas[-1])

    salida = Path("resultados/demo/pd" if a.demo else "resultados/pd")
    salida.mkdir(parents=True, exist_ok=True)
    tabla = pd.DataFrame(filas)
    (salida / "sensibilidad.md").write_text(tabla.to_markdown(index=False, disable_numparse=True) + "\n", encoding="utf-8")
    print(tabla.to_markdown(index=False))

    lista = df.assign(w=df["initial_list_status"].eq("w")).groupby("anio").agg(
        Creditos=("w", "size"), w=("w", "mean")).reset_index()
    lista = pd.DataFrame({"Año": lista["anio"], "Créditos": lista["Creditos"],
                          "Con initial_list_status = w": lista["w"].map("{:.1%}".format)})
    (salida / "lista_por_anio.md").write_text(lista.to_markdown(index=False, disable_numparse=True) + "\n",
                                              encoding="utf-8")
    print(lista.to_string(index=False))


if __name__ == "__main__":
    main()
