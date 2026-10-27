# HANDOFF: Trabajo 2 de Redes Neuronales (¿me prestas?)

Estado al 2026-10-26, noche. Repo local `D:\Dev\trabajo2-riesgo-credito` (todavía no está en GitHub).
Enunciado: `C:\Users\Juan Pablo\Downloads\Trabajo 2_ Redes neuronales para datos tabulares (¿me prestas.pdf`.
Fecha de entrega: Juan Pablo dijo que hay más tiempo que el 27-oct; falta confirmar la fecha exacta.

## Reparto del equipo

| Quién | Parte |
|---|---|
| Juan Pablo (A) | Datos (descarga, objetivo, variables, particiones, descriptivo con hipótesis) y Parte 2 completa (LGD, regresión lineal, red, pérdida esperada) |
| Persona B (por confirmar) | PD: logística, red, búsqueda, 30 semillas, AUC/Gini/KS/Brier/calibración, scorecard, interpretabilidad |
| Persona C (por confirmar) | App web (Streamlit), blog en GitHub Pages, fundamentación matemática, video promocional |
| Todos | `registro_ia.md` (prompts literales y alucinaciones), video individual, entender todo |

## Qué está hecho

- `preparar_datos.py`: de `loan.csv` (887 379 × 74, bajado de Kaggle el 2026-10-26) a
  `datos/procesados/pd.parquet` (268 530 créditos, 21.9 % malos) y `lgd.parquet` (46 009 castigados).
  Tarda 21 s.
- `descriptivo.py`: 10 hipótesis; 8 se confirman. H8 (verificado = menos riesgo) sale al revés:
  Verified 24.7 % vs Not Verified 17.1 %. H10 (antigüedad en el empleo) no muestra diferencia.
- `riesgo/` (paquete común): `configuracion.py` (config + `--set`), `datos.py`, `particiones.py`,
  `preparar.py`, `red.py` (MLP + bucle de entrenamiento con Adam y parada temprana), `metricas.py`.
- `lgd_entrenar.py` (media vs regresión lineal vs red con 30 semillas) y `lgd_busqueda.py`
  (búsqueda aleatoria de 12 configuraciones).

## Decisiones y por qué

- **D1. Objetivo PD:** la tabla del enunciado, en `riesgo/datos.py` (`MALO`). Current, Issued,
  In Grace Period y Late (16-30) quedan fuera.
- **D2. Periodo:** originación jun-2007 a dic-2015; pagos hasta ene-2016.
- **D3. Variables de entrada:** solo lo que se sabe al pedir el crédito. Se excluyen:
  - los pagos (`total_pymnt`, `recoveries`, `last_pymnt_d`, `out_prncp`...), porque son posteriores;
  - `grade`, `sub_grade`, `int_rate` e `installment` (por defecto), porque son el resultado del
    modelo de riesgo de Lending Club. Usarlos sería copiar otro modelo. Están en el parquet por si
    el equipo quiere compararlos;
  - `tot_cur_bal`, `tot_coll_amt` y `total_rev_hi_lim`, que solo existen desde 2012: el modelo
    aprendería la fecha a través de si están vacías;
  - los campos `*_joint` y los `open_*`/`il_*` (casi 100 % vacíos).
- **D4. Faltantes que significan algo:** `mths_since_last_delinq` vacío = nunca ha estado en mora.
  Se guarda en `nunca_en_mora` (y `sin_registros_publicos`) antes de imputar con la mediana.
- **D5. Partición PD:** entrenamiento y validación con originación ≤ 2013 (80/20 al azar,
  estratificado); prueba temporal 2014-2015. Ojo: en 2014-2015 solo hay desenlace para los que
  terminaron rápido (pagaron antes o cayeron pronto), así que la prueba tiene sesgo de madurez.
  Discutirlo en el reporte.
- **D6. LGD:** solo Charged Off (y "Does not meet... Charged Off"). Default y Late (31-120) tienen
  recoveries = 0 en el 100 % de los casos porque el cobro no ha terminado.
  - EAD = funded_amnt − total_rec_prncp (capital pendiente al castigo).
  - LGD = 1 − (recoveries − collection_recovery_fee) / EAD, recortada a [0, 1].
  - Limitaciones: no se descuenta el valor del dinero en el tiempo (no hay fechas de recuperación),
    no incluye intereses ni costos internos.
- **D7. LGD solo hasta 2013:** los castigos recientes aún no recuperan (en 2015 el 89 % tiene
  LGD = 1, contra 25 % en 2010-2012). Entrenamiento ≤ 2012, prueba 2013.
- **D8. Red:** PyTorch con bucle escrito a mano (se ve cada paso de la retropropagación y de Adam).
  Salida sigmoide en las dos tareas porque PD y LGD están entre 0 y 1. Pérdida de la LGD por
  defecto: entropía cruzada binaria con objetivo fraccional (cuasi-verosimilitud de Bernoulli,
  Papke y Wooldridge, 1996); `--set lgd.perdida=mse` para comparar.
- **D9. Preprocesamiento:** ajustado solo con entrenamiento (`riesgo/preparar.py`); log(1+x) en
  ingreso, saldo revolvente y monto; one-hot con categorías de menos de 50 casos agrupadas.

## Qué falta (parte A)

1. Correr `lgd_busqueda.py`, pasar la mejor configuración a `lgd.red` y volver a correr `lgd_entrenar.py`.
2. `perdida_esperada.py`: necesita el modelo de PD de la persona B. EAD para la pérdida esperada al
   originar = monto del préstamo (es lo que se expone el día uno); discutir.
3. Revisar en el diccionario y citar cada variable (`is_inc_v` es el nombre de `verification_status`).
4. Texto del blog: datos, caso de uso (con C), hipótesis, LGD.
5. Crear el repo en GitHub con el equipo y que todos hagan commits.

## Cómo correr

```
python preparar_datos.py        # solo si cambia algo de los datos (21 s)
python descriptivo.py           # hipótesis y figuras (10 s)
python lgd_entrenar.py --demo   # < 30 s
python lgd_entrenar.py          # 30 semillas
python lgd_busqueda.py          # búsqueda de hiperparámetros de la LGD
```

Python 3.14 en esta máquina; versiones en `requirements.txt`.
