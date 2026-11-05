# HANDOFF: Trabajo 2 de Redes Neuronales (¿me prestas?)

Estado al 2026-11-03. Repo `D:\Dev\trabajo2-riesgo-credito` → https://github.com/jperezalv22/trabajo2-riesgo-credito
(público; srodriguezsu y SATVanegas ya son colaboradores).
Enunciado: `enunciado/enunciado.pdf` (en el repo).
**Entrega: esta semana (antes del domingo 8 de noviembre); falta la fecha exacta.**

## Reparto del equipo

| Quién | Parte |
|---|---|
| Juan Pablo (A) | Datos (descarga, objetivo, variables, particiones, descriptivo con hipótesis) y Parte 2 completa (LGD, regresión lineal, red, pérdida esperada) |
| Santiago (B) | PD: logística, red, búsqueda, 30 semillas, AUC/Gini/KS/Brier/calibración, scorecard, interpretabilidad |
| C (el otro) | App web (Streamlit), blog en GitHub Pages, fundamentación matemática, video promocional |
| Todos | `registro_ia.md` (prompts literales y alucinaciones), video individual, entender todo |

## Notas para todo el equipo

- La app y la pérdida esperada predicen la red con `riesgo/red_numpy.py` (NumPy, sin PyTorch);
  los pesos se exportan con `riesgo.red.exportar_numpy` al entrenar.
- Diccionario de variables en `datos/diccionario/LCDataDictionary.xlsx` (sí está en el repo).

## Qué está hecho

- `preparar_datos.py`: de `loan.csv` (887 379 × 74, bajado de Kaggle el 2026-10-26) a
  `datos/procesados/pd.parquet` (268 530 créditos, 21.9 % malos) y `lgd.parquet` (46 009 castigados).
  Tarda 21 s.
- `descriptivo.py`: 10 hipótesis; 8 se confirman. H8 (verificado = menos riesgo) sale al revés:
  Verified 24.7 % vs Not Verified 17.1 %. H10 (antigüedad en el empleo) no muestra diferencia.
  También genera todas las tablas de las secciones 2, 3 y 6 del blog (estados, por_anio, variables,
  particiones_pd/lgd, lgd_por_anio, recuperaciones, verificacion) desde `estados_por_anio.csv`,
  que hace `preparar_datos.py`.
- Borradores del blog de la parte A (2026-11-03): `reporte/secciones/02_datos.md`,
  `03_descriptivo_hipotesis.md` y `06_lgd.md`. Numeración provisional: Tablas 1-12, Figuras 1-4.
  Cada uno termina con sus referencias en APA (verificadas, ver `registro_ia.md`).
- **D12. H8 y la app:** la verificación depende del monto (5.3 % sin verificar en el quintil más
  alto, 51.9 % en el más bajo), pero dentro de cada quintil los verificados siguen siendo más
  riesgosos. Es selección de Lending Club, no causa: el "¿qué pasaría si...?" de la app no debe
  sugerir cambiar `verification_status` para subir el puntaje. Avisarle a C.
- `riesgo/` (paquete común): `configuracion.py` (config + `--set`), `datos.py`, `particiones.py`,
  `preparar.py`, `red.py` (MLP + bucle de entrenamiento con Adam y parada temprana, `exportar_numpy`),
  `red_numpy.py` (predicción sin PyTorch, probada: diferencia máxima 1.2e-7 con PyTorch), `metricas.py`.
- `lgd_entrenar.py` (media vs regresión lineal vs red con 30 semillas) y `lgd_busqueda.py`
  (búsqueda aleatoria de 12 configuraciones).
- **D11. Red de la LGD = la ganadora de la búsqueda** (decidido el 2026-11-03): [64, 32], tanh,
  sin dropout, L2 1e-5, tasa 1e-3. Pérdida de validación 0.29578, contra 0.29635 ± 0.00017 de la
  anterior ([32, 16], relu, dropout 0.1). En la prueba 2013 sale algo peor: MAE 0.0763 ± 0.0006 y
  RMSE 0.0937, contra 0.0757 ± 0.0001 y 0.0930 de la anterior. Se deja igual porque se elige por
  validación; volver a la otra por la prueba sería usar la prueba para elegir. Para el blog: la
  validación (≤ 2012) no anticipa bien 2013, y sin dropout varía más entre semillas. Tarda 44 s
  en 30 semillas (la anterior, 142 s). Ninguna red le gana a la regresión lineal (MAE 0.0756).

## Decisiones y por qué

- **D1. Objetivo PD:** la tabla del enunciado, en `riesgo/datos.py` (`MALO`). Current, Issued,
  In Grace Period y Late (16-30) quedan fuera.
- **D2. Periodo:** originación jun-2007 a dic-2015; pagos hasta ene-2016.
- **D3. Variables de entrada:** solo lo que se sabe al pedir el crédito. Se excluyen:
  - los pagos (`total_pymnt`, `recoveries`, `last_pymnt_d`, `out_prncp`...), porque son posteriores;
  - `grade`, `sub_grade`, `int_rate` e `installment` (por defecto), porque son el resultado del
    modelo de riesgo de Lending Club. Usarlos sería copiar otro modelo. Están en el parquet por si
    el equipo quiere compararlos;
  - `tot_cur_bal`, `tot_coll_amt` y `total_rev_hi_lim`: vacías en el 100 % hasta 2011, en el 52 %
    en 2012 y completas desde 2013; el modelo aprendería la fecha a través de si están vacías;
  - los campos `*_joint` (99.9 % vacíos) y los `open_*`/`il_*` (97.6 % vacíos).
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
  - Hallazgo sin verificar (corregido el 2026-11-03): hasta 2013 la recuperación BRUTA / EAD,
    redondeada a 2 decimales, es 0 en el 38.9 %, 0.14 en el 24.7 % y 0.06 en el 11.2 %. No es
    "exactamente 0.14": antes se dijo así por mirar valores redondeados (candidato a la cacería, IA
    afirmando un dato con más precisión de la que tenía). La comisión de cobro es ~1 % o ~18 % de lo
    recuperado. Lo de una regla fija (¿venta de cartera?) sigue sin fuente primaria.
- **D7. LGD solo hasta 2013:** los castigos recientes aún no recuperan (en 2015 el 89 % tiene
  LGD = 1, contra 25 % en 2010-2012). Entrenamiento ≤ 2012, prueba 2013.
- **D8. Red:** PyTorch con bucle escrito a mano (se ve cada paso de la retropropagación y de Adam).
  Salida sigmoide en las dos tareas porque PD y LGD están entre 0 y 1. Pérdida de la LGD por
  defecto: entropía cruzada binaria con objetivo fraccional (cuasi-verosimilitud de Bernoulli,
  Papke y Wooldridge, 1996); `--set lgd.perdida=mse` para comparar.
- **D9. Preprocesamiento:** ajustado solo con entrenamiento (`riesgo/preparar.py`); log(1+x) en
  ingreso, saldo revolvente y monto; one-hot con categorías de menos de 50 casos agrupadas.
- **D10. Consola de Windows:** nada de símbolos fuera de cp1252 (≥, ≤, −) en lo que se imprime;
  `lgd_entrenar.py` se caía al imprimir la columna "≥ 0.99" (ahora ">= 0.99").

## Parte B (PD), estado al 2026-11-05

Hecho en la rama `parte-b-pd`: `pd_entrenar.py`, `pd_busqueda.py`, `pd_figuras.py`,
`pd_interpretar.py`, `pd_sensibilidad.py`, `pd_verificar.py`, `riesgo/scorecard.py`,
`riesgo/predecir_pd.py`, métricas de clasificación en `riesgo/metricas.py`, modelos en `modelos/`,
resultados en `resultados/pd/` y `reporte/secciones/05_pd.md` (Tablas P1 a P15 y Figuras P1 a P4).

Resultado en la prueba 2014-2015: logística AUC 0.6797 y Brier 0.1795 (90 parámetros, 2 s). Red AUC
0.6878 ± 0.0012 y Brier 0.1781 ± 0.0003 (19 841 parámetros, 762 s entre búsqueda y semillas). La
red gana en las 30 semillas, por poco.

Decisiones de la parte B (D1 a D12 no se cambiaron):

- **D13. Optimizador en la búsqueda:** el enunciado lo pide, así que `riesgo/red.py` lo acepta en
  `parametros` (Adam si no está). Adam y RMSprop comparten rango de tasas; SGD necesitaría el suyo.
  La LGD no cambia: su demo da los mismos números.
- **D14. Red de la PD = la ganadora de la búsqueda:** [128, 64], ReLU, dropout 0.3, L2 1e-5, tasa
  3e-4, lote 256, Adam. Le saca 0.00008 a la segunda ([64], 5825 parámetros), dentro del ruido entre
  semillas (0.00024). No se cambia porque se elige por validación.
- **D15. Escala de la scorecard:** 600 puntos = odds de 4:1 (PD de 20 %, cerca del promedio de
  entrenamiento) y PDO de 20. Solo cambia la escala: no mueve el AUC ni el percentil.
- **D16. Puntos por variable:** frente al cliente promedio de entrenamiento. Con la logística salen
  de los coeficientes; con la red, de valores de Shapley por muestreo (200 órdenes, semilla fija),
  porque la suma cuadra exactamente con el puntaje y solo necesita predecir. En un modelo lineal da
  lo mismo que los coeficientes (`pd_verificar.py`). `--set scorecard.modelo=logistica` cambia el
  modelo de la app.
- **D17. Sin recalibrar con la prueba:** la red predice 21.8 % y la tasa es 26.0 %, pero sería usar
  la prueba para ajustar y traería el sesgo de madurez (D5) como si fuera riesgo real.
- **D18. `initial_list_status` se queda:** `w` pasa de 0 % hasta 2011 a más de 50 % desde 2014, un
  problema parecido al que sacó otras variables en D3. Quitarla no ayuda en validación (AUC 0.6881 a
  0.6878) y sí en prueba (0.6797 a 0.6814), y no se decide por la prueba. Es candidata a salir si
  se reentrena con datos más recientes.
- **D19. Figuras sin reentrenar:** `pd_figuras.py` y `pd_interpretar.py` cargan `modelos/` y no
  usan PyTorch, como pide el enunciado.

Para C (app y blog):

- `evaluar_cliente` devuelve las llaves acordadas (`pd`, `puntaje`, `percentil`, `puntos`) más
  `puntos_base`, el puntaje del cliente de referencia. Se cumple que
  `puntos_base + suma(puntos) = puntaje`. Tarda unos 17 ms.
- Para el "¿qué pasaría si...?", llamar de nuevo a `evaluar_cliente` con el valor cambiado.
- Si no se conoce `nunca_en_mora` o `sin_registros_publicos`, se deducen de `meses_desde_mora` y
  `meses_desde_registro` (None = nunca). Si falta otra variable se imputa con la mediana y sale
  un aviso. Una categoría desconocida queda en ceros, sin error.
- `percentil` es el % de la población de entrenamiento con un puntaje menor o igual: más alto es mejor.
- La app no necesita torch ni matplotlib: se probó en un entorno sin ellos.
- Se recuerda D12: el "¿qué pasaría si...?" no debe sugerir cambiar `verification_status`.
- Sección 4: la L2 del código es `weight_decay`, que suma λθ al gradiente de todos los parámetros,
  sesgos incluidos. Equivale a (λ/2)‖θ‖² y no es AdamW. El optimizador final es Adam.
- Renumerar las Tablas P1 a P15 y las Figuras P1 a P4.

Para Juan Pablo (A), hallazgos en su parte que no se tocaron (`registro_ia.md`, A2 y A3):

- `descriptivo.py`: `rank(method="first")` reparte los empates por orden de archivo, y ese orden va
  con la fecha. En H4, los quintiles 1 y 2 son todos 0 y dan 18.3 % y 20.1 %; la tasa real con 0
  consultas es 19.9 %. Las conclusiones de H4 y H10 no cambian, pero sí la cifra y la frase "de forma
  ordenada" de la sección 3.2.
- Sección 3.3: dentro de año y plazo, la brecha de H8 baja de 7.6 a 2.9 puntos, y el modelo le da
  +0.1. Conviene revisar *"Para un modelo predictivo, "verificado" sí indica más riesgo"*.
- La Figura 4 usa terracota y verde, que casi no se distinguen con daltonismo protán. Es opcional.
- Pérdida esperada: la PD se queda corta en 2014-2015 (21.8 % contra 26.0 %), así que la pérdida
  esperada en esos años también lo hará.

## Qué falta (parte A)

1. `perdida_esperada.py`: ya puede usar `riesgo/predecir_pd.py` (`probabilidad(modelos, df)`), en la
   rama `parte-b-pd`. EAD para la pérdida esperada al originar = monto del préstamo (es lo que se
   expone el día uno); discutir.
2. Sección 07_perdida_esperada (cuando esté la PD). Las secciones 02, 03 y 06 ya tienen borrador:
   falta que Juan Pablo las lea y ajuste lo que no le suene a él.
3. Fuente primaria para las fracciones de recuperación fijas (D6): no se encontró; en el blog
   queda como pregunta abierta.
4. Revisar y fusionar los pull requests de B y C; activar GitHub Pages desde `docs/` (solo el dueño
   puede) y, si hace falta, conectar Streamlit Community Cloud al repo.
5. Probar todo en un clon limpio y grabar el video individual.

## Cómo correr

```
python preparar_datos.py        # solo si cambia algo de los datos (21 s)
python descriptivo.py           # hipótesis, tablas y figuras del blog (15 s)
python lgd_entrenar.py --demo   # < 30 s
python lgd_entrenar.py          # 30 semillas
python lgd_busqueda.py          # búsqueda de hiperparámetros de la LGD
python pd_entrenar.py --demo    # < 15 s
python pd_entrenar.py           # logística y red con 30 semillas (9 min)
python pd_busqueda.py           # búsqueda de hiperparámetros de la PD (4 min)
python pd_figuras.py            # figuras y tablas sin reentrenar (10 s)
python pd_interpretar.py        # importancia de variables e hipótesis (25 s)
python pd_sensibilidad.py       # pesos por clase y variables (15 s)
python pd_verificar.py          # comprobaciones contra resultados conocidos (10 s)
```

Python 3.14 en esta máquina; versiones en `requirements.txt`.
