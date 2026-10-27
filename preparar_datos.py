"""Convierte datos/original/loan/loan.csv en datos/procesados/pd.parquet y lgd.parquet.

Solo hace falta correrlo una vez: los procesados ya estan en el repositorio.
El original (442 MB) se baja de Kaggle, ver README.

    python preparar_datos.py
"""

import json
from pathlib import Path

from riesgo import configuracion, datos


def main():
    a = configuracion.argumentos(__doc__).parse_args()
    cfg = configuracion.cargar(a.config, a.cambios)
    pd_tabla, lgd_tabla = datos.procesar(cfg["datos"]["original"], cfg["datos"]["procesados"])

    resumen = {
        "pd_registros": len(pd_tabla),
        "pd_malos": int(pd_tabla["malo"].sum()),
        "pd_tasa_malos": round(pd_tabla["malo"].mean(), 4),
        "lgd_registros": len(lgd_tabla),
        "lgd_media": round(lgd_tabla["lgd"].mean(), 4),
        "lgd_igual_a_1": round((lgd_tabla["lgd"] == 1).mean(), 4),
        "originacion": f"{pd_tabla['fecha'].min():%Y-%m} a {pd_tabla['fecha'].max():%Y-%m}",
    }
    salida = Path(cfg["datos"]["procesados"]) / "resumen.json"
    salida.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resumen, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
