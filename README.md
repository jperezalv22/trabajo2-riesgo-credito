# Trabajo 2: Redes neuronales para datos tabulares (¿me prestas?)

Redes neuronales y algoritmos bioinspirados, Universidad Nacional de Colombia, sede Medellín, 2026-02.

Un banco quiere saber, antes de prestar, qué tan probable es que no le paguen (PD, clasificación)
y, si no le pagan, qué fracción del dinero pierde (LGD, regresión). Comparamos redes neuronales
con modelos simples bajo las mismas condiciones.

El enunciado del profesor está en [`enunciado/enunciado.pdf`](enunciado/enunciado.pdf).

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
| Análisis descriptivo, hipótesis y tablas del blog | `python descriptivo.py` | 20 s |
| LGD, demostración | `python lgd_entrenar.py --demo` | 25 s |
| LGD completa (30 semillas) | `python lgd_entrenar.py` | 1 min |
| LGD, búsqueda de hiperparámetros | `python lgd_busqueda.py` | 1 min |
| PD, demostración | `python pd_entrenar.py --demo` | 10 s |
| PD completa (logística y red con 30 semillas) | `python pd_entrenar.py` | 10 min |
| PD, búsqueda de hiperparámetros | `python pd_busqueda.py` | 4 min |
| PD, figuras y tablas sin reentrenar | `python pd_figuras.py` | 10 s |
| PD, importancia de variables e hipótesis | `python pd_interpretar.py` | 25 s |
| PD, sensibilidad (pesos por clase y variables) | `python pd_sensibilidad.py` | 15 s |
| PD, comprobaciones contra resultados conocidos | `python pd_verificar.py` | 10 s |
| Pérdida esperada de cada crédito (usa los modelos guardados) | `python perdida_esperada.py` | 12 s |
| Aplicación web (abre en el navegador) | `streamlit run app/app.py` | 5 s |
| Blog: une las secciones y genera `docs/index.html` | `python construir_blog.py` | 3 s |

Todos los scripts de la PD aceptan `--demo` (muestra del 10 %, menos de 2 minutos), salvo
`pd_verificar.py`, que no entrena. Las salidas de demostración van a `resultados/demo/` y no tocan
`modelos/` ni `resultados/pd/`.

Ejemplos de cambio de parámetros:

```bash
python lgd_entrenar.py --demo --set lgd.perdida=mse --set lgd.red.capas=[64,32]
python pd_entrenar.py --demo --set pd.red.capas=[32,16] --set pd.red.dropout=0
python pd_entrenar.py --demo --set scorecard.pdo=40 --set scorecard.puntaje_base=700
```

## Aplicación web

```bash
streamlit run app/app.py
```

La persona llena sus datos (vienen llenos con los de una persona típica: la mediana o la categoría
más común de `datos/procesados/pd.parquet`) y ve su puntaje, su probabilidad de no pagar, su percentil
frente a la población, qué le suma y qué le resta puntos, y qué pasaría si cambiara una característica.
Los enlaces al blog y al video, y las variables que el "¿qué pasaría si...?" no ofrece cambiar, están
en la sección `app` de `config.yaml`.

Para usar solo la app no hace falta PyTorch: `pip install -r app/requirements.txt`.

**Publicación en Streamlit Community Cloud:** en https://share.streamlit.io, "Create app" desde este
repositorio, rama `main`, archivo `app/app.py`. Community Cloud usa `app/requirements.txt` (busca
primero en la carpeta del archivo de entrada). En "Advanced settings", Python 3.12 o más nuevo.

## Blog

```bash
python construir_blog.py
```

Une `reporte/secciones/*.md` en `reporte/blog.md` y lo convierte en `docs/index.html` con
[pandoc](https://pandoc.org/installing.html) (fórmulas con MathJax). Pone la numeración final de
figuras y tablas (cada sección usa una provisional, como "Tabla P3"), cambia las menciones del texto y
avisa si una figura o tabla no se cita, si se cita una que no existe o si una referencia no está en
`11_referencias.md`. Sin pandoc, igual genera `reporte/blog.md`.

GitHub Pages publica la carpeta `docs/` de `main` (Settings > Pages > Deploy from a branch > `main` /
`docs`).

## Predicción sin PyTorch

La app y la pérdida esperada usan `riesgo/predecir_pd.py`, que no importa PyTorch:

```python
from riesgo import predecir_pd

modelos = predecir_pd.cargar("modelos")
predecir_pd.probabilidad(modelos, df)          # df con las columnas de datos/procesados/pd.parquet
predecir_pd.evaluar_cliente(modelos, cliente)  # {"pd", "puntaje", "percentil", "puntos", "puntos_base"}
```

La aplicación no necesita instalar PyTorch ni matplotlib.

## Semillas

Semilla base 2026 (`config.yaml`). La corrida k de la red usa la semilla 2026 + k.
