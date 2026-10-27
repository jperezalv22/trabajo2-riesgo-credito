# Trabajo 2: Redes neuronales para datos tabulares (¿me prestas?)

Redes neuronales y algoritmos bioinspirados, Universidad Nacional de Colombia, sede Medellín, 2026-02.

Un banco quiere saber, antes de prestar, qué tan probable es que no le paguen (PD, clasificación)
y, si no le pagan, qué fracción del dinero pierde (LGD, regresión). Comparamos redes neuronales
con modelos simples bajo las mismas condiciones.

> En construcción. El reparto del trabajo y las decisiones están en [`HANDOFF.md`](HANDOFF.md).

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate            # en Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## Datos

Credit Risk Analysis (Lending Club), https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset,
descargado el 2026-10-26: 887 379 créditos y 74 variables, originados entre junio de 2007 y
diciembre de 2015.

Los datos procesados ya están en `datos/procesados/`, así que **no hace falta internet** para
correr los modelos. Para regenerarlos, baje `loan.csv` y `LCDataDictionary.xlsx` de Kaggle en
`datos/original/` y corra `python preparar_datos.py`.

## Cómo correr

Todos los parámetros están en `config.yaml` y se pueden cambiar con `--set`, sin editar el código.

| Qué | Comando | Tiempo aprox. |
|---|---|---|
| Datos procesados desde el original | `python preparar_datos.py` | 21 s |
| Análisis descriptivo e hipótesis | `python descriptivo.py` | 10 s |
| LGD, demostración | `python lgd_entrenar.py --demo` | 25 s |
| LGD completa (30 semillas) | `python lgd_entrenar.py` | 3 min |
| LGD, búsqueda de hiperparámetros | `python lgd_busqueda.py` | 1 min |

Ejemplo de cambio de parámetros:

```bash
python lgd_entrenar.py --demo --set lgd.perdida=mse --set lgd.red.capas=[64,32]
```

## Semillas

Semilla base 2026 (`config.yaml`). La corrida k de la red usa la semilla 2026 + k.
