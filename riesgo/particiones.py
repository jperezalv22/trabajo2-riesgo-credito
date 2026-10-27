"""Particiones de entrenamiento, validacion y prueba.

Prueba es temporal: los creditos originados despues de 'ultimo_anio_entrenamiento'.
Asi se mide lo que pasaria en la practica, donde el modelo se entrena con el
pasado y se usa con clientes nuevos.

Validacion sale al azar de los anios de entrenamiento (estratificada por la
variable objetivo) y se usa para la parada temprana y para elegir hiperparametros.
"""

from sklearn.model_selection import train_test_split


def dividir(df, ultimo_anio, fraccion_validacion, semilla, estratificar=None):
    """Devuelve (entrenamiento, validacion, prueba) como DataFrames."""
    pasado = df[df["anio"] <= ultimo_anio]
    prueba = df[df["anio"] > ultimo_anio]
    estratos = pasado[estratificar] if estratificar else None
    entrenamiento, validacion = train_test_split(
        pasado, test_size=fraccion_validacion, random_state=semilla, stratify=estratos)
    return entrenamiento, validacion, prueba


def resumen(entrenamiento, validacion, prueba, objetivo):
    """Tamano, anios y promedio del objetivo de cada particion (para el reporte)."""
    filas = []
    for nombre, parte in [("entrenamiento", entrenamiento), ("validacion", validacion),
                          ("prueba", prueba)]:
        filas.append({
            "particion": nombre,
            "registros": len(parte),
            "anios": f"{parte['anio'].min()}-{parte['anio'].max()}",
            f"{objetivo}_promedio": float(round(parte[objetivo].mean(), 4)),
        })
    return filas
