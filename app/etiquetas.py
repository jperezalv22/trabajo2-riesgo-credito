"""Textos de la aplicacion: cada variable del modelo con su pregunta en espanol sencillo.

Las ayudas son traducciones de las definiciones del diccionario del conjunto de datos
(datos/diccionario/LCDataDictionary.xlsx, hoja LoanStats). verification_status aparece alli
como is_inc_v.
"""

# variable: (pregunta, ayuda)
PREGUNTAS = {
    # sobre la persona
    "annual_inc": ("Ingreso anual (dólares)",
                   "El ingreso anual que usted declara al registrarse."),
    "antiguedad_empleo": ("Años en su empleo actual",
                          "0 es menos de un año y 10 es diez años o más."),
    "home_ownership": ("Vivienda", "Si su vivienda es propia, hipotecada o arrendada."),
    "addr_state": ("Estado donde vive (EE. UU.)",
                   "El estado que usted escribe en la solicitud."),
    "verification_status": ("Verificación del ingreso",
                            "Si Lending Club verificó su ingreso, verificó la fuente del "
                            "ingreso o no lo verificó."),
    # el prestamo
    "loan_amnt": ("Monto que pide (dólares)", "Cuánto dinero solicita prestado."),
    "plazo_meses": ("Plazo (meses)", "Número de cuotas: 36 o 60 meses."),
    "purpose": ("¿Para qué es el préstamo?", "La categoría que usted elige en la solicitud."),
    # el diccionario solo dice "Possible values are - W, F", sin explicar que significa cada una
    "initial_list_status": ("Estado inicial de publicación del préstamo",
                            "Lo asigna Lending Club al publicar el préstamo (W o F). No depende "
                            "de usted: déjelo como está."),
    # historial de credito
    "antiguedad_credito_meses": ("Antigüedad de su historial de crédito (años)",
                                 "Tiempo desde que abrió su primera línea de crédito."),
    "dti": ("Deudas mensuales sobre ingreso mensual (%)",
            "Lo que paga al mes en deudas (sin la hipoteca ni este préstamo) dividido por su "
            "ingreso mensual, en porcentaje."),
    "revol_bal": ("Saldo en tarjetas y créditos rotativos (dólares)",
                  "Lo que debe hoy en créditos rotativos, como tarjetas de crédito."),
    "revol_util": ("Uso del cupo rotativo (%)",
                   "Qué parte de su cupo rotativo disponible está usando."),
    "open_acc": ("Líneas de crédito abiertas", "Cuántas líneas de crédito tiene abiertas hoy."),
    "total_acc": ("Líneas de crédito en total",
                  "Cuántas líneas de crédito aparecen en su historial."),
    "inq_last_6mths": ("Consultas a su historial en los últimos 6 meses",
                       "Veces que una entidad consultó su crédito (sin contar carro ni "
                       "vivienda)."),
    "delinq_2yrs": ("Moras de más de 30 días en los últimos 2 años",
                    "Veces que se atrasó más de 30 días en un pago."),
    "nunca_en_mora": ("¿Alguna vez ha estado en mora?",
                      "Si en su historial aparece algún atraso en un pago."),
    "meses_desde_mora": ("Meses desde su última mora", "Cuánto hace de su último atraso."),
    "acc_now_delinq": ("Cuentas en mora hoy", "En cuántas cuentas está atrasado ahora."),
    "collections_12_mths_ex_med": ("Cobros de cartera en los últimos 12 meses",
                                   "Deudas pasadas a cobranza, sin contar las médicas."),
    "sin_registros_publicos": ("¿Tiene registros públicos negativos?",
                               "Si en su historial aparece algún registro público negativo."),
    "meses_desde_registro": ("Meses desde su último registro público",
                             "Cuánto hace del último."),
    "pub_rec": ("Número de registros públicos negativos",
                "Cuántos registros públicos negativos tiene."),
}

# En el formulario, las variables van en este orden y en estos grupos
GRUPOS = {
    "Sobre usted": ["annual_inc", "antiguedad_empleo", "home_ownership", "addr_state",
                    "verification_status"],
    "El préstamo": ["loan_amnt", "plazo_meses", "purpose", "initial_list_status"],
    "Su historial de crédito": ["antiguedad_credito_meses", "dti", "revol_bal", "revol_util",
                                "open_acc", "total_acc", "inq_last_6mths", "delinq_2yrs",
                                "nunca_en_mora", "meses_desde_mora", "acc_now_delinq",
                                "collections_12_mths_ex_med", "sin_registros_publicos",
                                "meses_desde_registro", "pub_rec"],
}

# Nombres de las categorias tal como vienen en los datos
CATEGORIAS = {
    "home_ownership": {"RENT": "Arrendada", "OWN": "Propia", "MORTGAGE": "Hipotecada",
                       "OTHER": "Otra", "NONE": "Ninguna", "ANY": "Cualquiera"},
    "verification_status": {"Verified": "Ingreso verificado",
                            "Source Verified": "Fuente del ingreso verificada",
                            "Not Verified": "No verificado"},
    "purpose": {"debt_consolidation": "Unificar deudas", "credit_card": "Pagar tarjetas de crédito",
                "home_improvement": "Mejoras a la vivienda", "other": "Otro",
                "major_purchase": "Compra grande", "small_business": "Pequeño negocio",
                "car": "Carro", "medical": "Gastos médicos", "moving": "Mudanza",
                "vacation": "Vacaciones", "house": "Vivienda", "wedding": "Boda",
                "renewable_energy": "Energía renovable", "educational": "Estudios"},
    "initial_list_status": {"f": "F", "w": "W"},
}


def pregunta(variable):
    return PREGUNTAS[variable][0]


def ayuda(variable):
    return PREGUNTAS[variable][1]


def categoria(variable, valor):
    """Nombre en espanol de una categoria; si no esta en la lista, el valor original."""
    return CATEGORIAS.get(variable, {}).get(valor, valor)
