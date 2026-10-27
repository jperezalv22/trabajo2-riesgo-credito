"""Lectura de config.yaml y cambios desde la linea de comandos.

Ejemplo:
    python lgd_entrenar.py --set lgd.red.capas=[64,32] --set lgd.perdida=mse
"""

import argparse

import yaml


def cargar(ruta="config.yaml", cambios=()):
    """Lee el YAML y aplica cambios del tipo 'seccion.parametro=valor'."""
    with open(ruta, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for cambio in cambios:
        clave, valor = cambio.split("=", 1)
        *ruta_claves, ultima = clave.split(".")
        nodo = cfg
        for c in ruta_claves:
            nodo = nodo[c]
        # yaml interpreta el valor: 0.5 -> float, [64,32] -> lista, mse -> texto
        nodo[ultima] = yaml.safe_load(valor)
    return cfg


def argumentos(descripcion):
    """Argumentos comunes a todos los scripts: config, cambios y modo demo."""
    p = argparse.ArgumentParser(description=descripcion,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--set", dest="cambios", action="append", default=[],
                   metavar="CLAVE=VALOR", help="cambia un parametro del config (se puede repetir)")
    p.add_argument("--demo", action="store_true",
                   help="usa una muestra de los datos para que corra en menos de 2 minutos")
    return p
