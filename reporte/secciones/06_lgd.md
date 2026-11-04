<!-- Sección 6. Autor: Juan Pablo Pérez Álvarez.
     Rutas de figuras relativas a reporte/. Numeración de figuras y tablas provisional.
     Tablas: `python descriptivo.py` (lgd_por_anio, recuperaciones, particiones_lgd),
     `python lgd_busqueda.py` (busqueda.md) y `python lgd_entrenar.py` (metricas.md, cuantiles.md). -->

## 6. Pérdida dado incumplimiento (LGD)

Cuando un cliente no paga, Lending Club castiga el crédito y después intenta recuperar parte del dinero. La pérdida dado incumplimiento (LGD) es la fracción de la exposición que al final se pierde: 0 si se recupera todo y 1 si no se recupera nada.

### 6.1 Variable objetivo

Para cada crédito castigado calculamos la exposición al incumplimiento (EAD) como el capital que faltaba por pagar, y la LGD como uno menos la fracción de esa exposición que se recuperó, neta de la comisión de cobro:

$$
\text{EAD}_i = \texttt{funded\_amnt}_i - \texttt{total\_rec\_prncp}_i
$$

$$
\text{LGD}_i = \min\left(1,\ \max\left(0,\ 1 - \frac{\texttt{recoveries}_i - \texttt{collection\_recovery\_fee}_i}{\text{EAD}_i}\right)\right)
$$

Cada campo tiene una razón, con su definición en el diccionario de Lending Club (ranadeep, 2021):

- `funded_amnt` (*"The total amount committed to that loan at that point in time"*) es lo que de verdad se prestó. No usamos `loan_amnt`, que es lo que se pidió y puede ser distinto.
- `total_rec_prncp` (*"Principal received to date"*) es el capital que la persona alcanzó a pagar antes de dejar de hacerlo. Restarlo deja el capital pendiente en el momento del castigo, que es lo que se expone.
- `recoveries` (*"post charge off gross recovery"*) es lo que se recuperó después del castigo, en bruto.
- `collection_recovery_fee` (*"post charge off collection fee"*) es la comisión que se pagó por cobrar. Se resta porque esa plata no le llega al prestamista.

Solo 38 de los 46 009 créditos castigados (0.08 %) recuperaron más que su exposición y tuvieron que recortarse a 0. Ninguno pasó de 1.

**Qué créditos entran.** Usamos solo los castigados: *Charged Off* y *Does not meet the credit policy. Status:Charged Off*, 46 009 créditos. Los que están en *Default* o *Late (31-120 days)* también son malos para la PD, pero el 100 % de ellos tiene `recoveries` igual a 0: el cobro todavía no ha empezado o no ha terminado, y su LGD saldría en 1 de forma artificial.

**Qué años entran.** La Tabla 7 muestra la LGD por año de originación. En los castigados de 2010 a 2012, entre el 23 % y el 26 % no recuperó nada (LGD = 1); en los de 2014 ese porcentaje sube al 62.7 % y en los de 2015, al 89.0 %. Un crédito originado en 2015 y castigado poco antes de enero de 2016 no ha tenido tiempo de recuperar nada, así que esa LGD de 1 no es la pérdida final sino el estado de un cobro a medias. Por eso dejamos por fuera 2014 y 2015, entrenamos con 2007 a 2012 y probamos con 2013. La Figura 3 compara las dos distribuciones.

**Tabla 7.** LGD de los créditos castigados por año de originación.

|   Año | Uso                            | Castigados   |   LGD media | LGD = 1   |
|------:|:-------------------------------|:-------------|------------:|:----------|
|  2007 | entrenamiento y validación     | 158          |       0.921 | 36.7%     |
|  2008 | entrenamiento y validación     | 496          |       0.931 | 44.2%     |
|  2009 | entrenamiento y validación     | 723          |       0.926 | 34.0%     |
|  2010 | entrenamiento y validación     | 1 753        |       0.925 | 25.6%     |
|  2011 | entrenamiento y validación     | 3 205        |       0.922 | 22.7%     |
|  2012 | entrenamiento y validación     | 8 084        |       0.911 | 25.2%     |
|  2013 | prueba                         | 14 836       |       0.918 | 42.9%     |
|  2014 | se excluye: cobro sin terminar | 13 981       |       0.945 | 62.7%     |
|  2015 | se excluye: cobro sin terminar | 2 773        |       0.981 | 89.0%     |

*Fuente: elaboración propia con datos de Lending Club (ranadeep, 2021).*

![Figura 3](../resultados/descriptivo/lgd_distribucion.png)

**Figura 3.** Distribución de la LGD de los créditos castigados, originados hasta 2013 (se usan) y después de 2013 (se excluyen). *Fuente: elaboración propia.*

El mismo efecto se asoma en 2013: el 42.9 % de sus castigados tiene LGD = 1, contra el 23-26 % de 2010-2012. Probablemente a algunos de esos créditos todavía les faltaba cobro en enero de 2016. Lo dejamos en la prueba porque es el año más reciente que todavía se puede usar, pero hay que tenerlo presente al leer los resultados: la prueba tiene más créditos sin recuperación que el entrenamiento.

**Una distribución con picos.** La Figura 3 tiene picos en lugares muy precisos: cerca de 0.86, cerca de 0.94 y en 1. La Tabla 8 muestra de dónde salen. Entre los castigados hasta 2013, el 38.9 % no recuperó nada, el 24.7 % recuperó cerca del 14 % de su exposición y el 11.2 %, cerca del 6 %. La comisión de cobro también se concentra en dos valores: alrededor del 1 % o del 18 % de lo recuperado.

**Tabla 8.** Valores más frecuentes de la recuperación bruta sobre la exposición y de la comisión de cobro sobre la recuperación (castigados originados hasta 2013).

|   Recuperación bruta / EAD (2 decimales) | % de los castigados   |
|-----------------------------------------:|:----------------------|
|                                     0    | 38.9%                 |
|                                     0.14 | 24.7%                 |
|                                     0.06 | 11.2%                 |
|                                     0.15 | 4.9%                  |
|                                     0.18 | 4.1%                  |

|   Comisión de cobro / recuperación (2 decimales) | % de los que recuperaron algo   |
|-------------------------------------------------:|:--------------------------------|
|                                             0.01 | 58.8%                           |
|                                             0.18 | 26.3%                           |
|                                             0    | 6.1%                            |
|                                             0.17 | 2.6%                            |

*Fuente: elaboración propia con datos de Lending Club (ranadeep, 2021). Los valores están redondeados a dos decimales.*

Que la recuperación caiga casi siempre en los mismos dos o tres valores sugiere que la decide una regla fija del proceso de cobro, como vender la cartera castigada a un precio por dólar, más que las características de cada persona. No encontramos una fuente primaria de Lending Club que lo confirme, así que lo dejamos como pregunta abierta. Para el modelo sí importa: si la recuperación depende del proceso de cobro, las variables de la solicitud tienen poco que decir sobre ella.

**Limitaciones de esta LGD.**

- No descuenta el valor del dinero en el tiempo. Un dólar recuperado tres años después vale menos que uno recuperado al mes, pero el conjunto no trae las fechas de las recuperaciones.
- La exposición es solo capital. No incluye los intereses que se dejaron de cobrar.
- Solo resta la comisión de cobro registrada; los costos internos de cobranza no aparecen en los datos.
- Aun hasta 2013 puede haber cobros sin terminar, como sugiere la Tabla 7.

### 6.2 Modelos

Comparamos tres modelos con las mismas particiones y el mismo preprocesamiento (sección 2.3, ajustado con el entrenamiento de la LGD). La Tabla 9 resume las particiones.

**Tabla 9.** Particiones para la LGD.

| Partición     | Créditos   | Años      |   LGD media |
|:--------------|:-----------|:----------|------------:|
| entrenamiento | 11 535     | 2007-2012 |       0.918 |
| validacion    | 2 884      | 2007-2012 |       0.912 |
| prueba        | 14 836     | 2013-2013 |       0.918 |

*Fuente: elaboración propia (`riesgo/particiones.py`, semilla 2026).*

1. **La media.** Predice para todos la LGD promedio del entrenamiento. Es el piso: cualquier modelo útil debería ganarle.
2. **Regresión lineal** por mínimos cuadrados, el modelo de baja complejidad. Tiene 78 parámetros: un coeficiente por cada una de las 77 columnas que salen del preprocesamiento, más el intercepto.
3. **Red neuronal.** Un perceptrón multicapa con una neurona de salida con activación sigmoide, entrenado con un bucle escrito a mano en PyTorch (`riesgo/red.py`): Adam, lotes de 512, hasta 60 épocas y parada temprana cuando la pérdida en validación no mejora en 6 épocas seguidas. Al final se recuperan los pesos de la mejor época.

**Activación de salida y función de pérdida.** La LGD está entre 0 y 1, así que usamos la sigmoide en la salida: cualquier predicción queda en ese rango. Como pérdida usamos la entropía cruzada binaria con el objetivo fraccional $y_i \in [0, 1]$:

$$
\mathcal{L} = -\frac{1}{n} \sum_{i=1}^{n} \left[\, y_i \log \hat{y}_i + (1 - y_i) \log (1 - \hat{y}_i) \,\right]
$$

Aunque $y_i$ no sea 0 o 1, esta pérdida sirve para estimar la media de la LGD. Si para unas características $x$ la LGD esperada es $m = \mathbb{E}[y \mid x]$, la pérdida esperada de predecir $p$ es $-m \log p - (1 - m) \log (1 - p)$. Su derivada, $-m/p + (1 - m)/(1 - p)$, se anula en $p = m$. Papke y Wooldridge (1996) proponen justamente esta cuasi-verosimilitud de Bernoulli para variables que son fracciones, y muestran que da estimaciones consistentes de la media condicional siempre que la forma de esa media esté bien especificada. El error cuadrático también se minimiza en $p = m$. Preferimos la entropía cruzada por cómo se comporta el gradiente: respecto a la entrada $z$ de la sigmoide vale $\hat{y} - y$, mientras que con el error cuadrático vale $2(\hat{y} - y)\,\hat{y}(1 - \hat{y})$, que se hace casi cero cuando la sigmoide se satura cerca de 0 o de 1. La derivación desde la verosimilitud está en la sección 4. Con `--set lgd.perdida=mse` se puede entrenar la misma red con error cuadrático y comparar.

### 6.3 Búsqueda de hiperparámetros

La rejilla de `config.yaml` combina cinco arquitecturas ([16], [32], [32, 16], [64, 32] y [64, 32, 16] neuronas en las capas ocultas), dos activaciones (ReLU y tanh), tres valores de dropout (0, 0.1 y 0.3), tres de penalización L2 (0, $10^{-5}$ y $10^{-4}$) y tres tasas de aprendizaje ($3 \cdot 10^{-4}$, $10^{-3}$ y $3 \cdot 10^{-3}$), en total 270 combinaciones. En vez de probarlas todas, sorteamos 12 sin repetir, porque con el mismo número de intentos una búsqueda aleatoria explora más valores distintos de cada hiperparámetro que una rejilla (Bergstra y Bengio, 2012). Cada configuración se entrenó una vez con la semilla 2026 y se ordenaron por la pérdida en validación. La prueba no se usó para elegir.

El presupuesto fue de 12 configuraciones y 46 segundos de entrenamiento en total. La regresión lineal no tiene hiperparámetros: se ajustó una vez, en 0.1 segundos. La Tabla 10 muestra los 12 intentos.

**Tabla 10.** Búsqueda aleatoria de hiperparámetros de la red de la LGD, ordenada por pérdida en validación.

|   Intento | Capas ocultas   | Activación   |   Dropout |     L2 |   Tasa de aprendizaje |   Pérdida en validación |   MAE en validación |   Parámetros |   Segundos |
|----------:|:----------------|:-------------|----------:|-------:|----------------------:|------------------------:|--------------------:|-------------:|-----------:|
|        10 | [64, 32]        | tanh         |       0   | 1e-05  |                0.001  |                 0.29578 |              0.0776 |         7105 |        1.8 |
|         7 | [64, 32]        | tanh         |       0.1 | 1e-05  |                0.001  |                 0.29582 |              0.0774 |         7105 |        3.7 |
|         9 | [64, 32, 16]    | tanh         |       0   | 1e-05  |                0.0003 |                 0.29593 |              0.078  |         7617 |        5.8 |
|         5 | [16]            | tanh         |       0.3 | 0.0001 |                0.0003 |                 0.29608 |              0.0786 |         1265 |        3.8 |
|         6 | [64, 32]        | tanh         |       0.3 | 0      |                0.001  |                 0.29609 |              0.0769 |         7105 |        3.2 |
|         2 | [32, 16]        | relu         |       0.1 | 1e-05  |                0.001  |                 0.29614 |              0.0769 |         3041 |        6.2 |
|         4 | [16]            | tanh         |       0.1 | 0      |                0.0003 |                 0.29615 |              0.0788 |         1265 |        2.6 |
|         1 | [64, 32, 16]    | relu         |       0   | 1e-05  |                0.001  |                 0.29632 |              0.0776 |         7617 |        5.5 |
|        12 | [32, 16]        | relu         |       0.1 | 1e-05  |                0.0003 |                 0.29635 |              0.0777 |         3041 |        6.6 |
|         3 | [64, 32, 16]    | tanh         |       0.3 | 0.0001 |                0.003  |                 0.2966  |              0.0801 |         7617 |        1.2 |
|        11 | [64, 32, 16]    | relu         |       0   | 0      |                0.001  |                 0.2966  |              0.0777 |         7617 |        2.1 |
|         8 | [32]            | relu         |       0   | 0      |                0.0003 |                 0.2967  |              0.0778 |         2529 |        3.9 |

*Fuente: elaboración propia con `python lgd_busqueda.py`.*

Las 12 configuraciones quedaron muy cerca: la pérdida en validación va de 0.29578 a 0.29670, una diferencia de 0.3 %. Ganó la red de dos capas ocultas de 64 y 32 neuronas con tanh, sin dropout, L2 de $10^{-5}$ y tasa de aprendizaje de $10^{-3}$, que tiene 7105 parámetros. Esa es la configuración final que está en `lgd.red` de `config.yaml`.

Antes de la búsqueda usábamos otra red ([32, 16] con ReLU y dropout de 0.1, el intento 2, en el sexto lugar). Al entrenar la ganadora con 30 semillas resultó un poco peor en la prueba que la anterior: error absoluto medio de 0.0763 contra 0.0757. La dejamos igual. Volver a la anterior porque le fue mejor en la prueba sería usar la prueba para elegir, y entonces la prueba dejaría de medir lo que pasaría con datos nuevos. La lección es que la validación, sacada al azar de 2007 a 2012, no anticipa bien lo que pasa en 2013.

### 6.4 Resultados

La red final se entrenó con 30 semillas distintas (2026 a 2055). La Tabla 11 compara los tres modelos en la prueba de 2013.

**Tabla 11.** Error de los modelos de LGD en la prueba (créditos castigados originados en 2013). Para la red: media ± desviación estándar en 30 semillas, con la mejor y la peor.

| Modelo            | MAE                                         | RMSE                                        |   Parámetros |   Corridas |   Segundos |
|:------------------|:--------------------------------------------|:--------------------------------------------|-------------:|-----------:|-----------:|
| Media (constante) | 0.0764                                      | 0.0925                                      |            1 |          1 |        0   |
| Regresión lineal  | 0.0756                                      | 0.0927                                      |           78 |          1 |        0.1 |
| Red neuronal      | 0.0763 ± 0.0006 (mejor 0.0754, peor 0.0779) | 0.0937 ± 0.0011 (mejor 0.0927, peor 0.0977) |         7105 |         30 |       44   |

*Fuente: elaboración propia con `python lgd_entrenar.py`. MAE: error absoluto medio; RMSE: raíz del error cuadrático medio (sección 4).*

Ningún modelo le gana con claridad a la media. La regresión lineal mejora el MAE en menos de una milésima (0.0756 contra 0.0764) y la red, en promedio, en una diezmilésima (0.0763). En RMSE la media es la mejor de las tres (0.0925). La mejor de las 30 semillas de la red llega a 0.0754 de MAE, pero esa semilla no se puede escoger de antemano: lo que se puede esperar de la red es la media, 0.0763, con una desviación de 0.0006.

La Tabla 12 y la Figura 4 comparan la distribución de las predicciones con la observada.

**Tabla 12.** Cuantiles de la LGD observada y de las predicciones en la prueba.

| Distribución     |   p10 |   p25 |   p50 |   p75 |   p90 | Fuera de [0, 1]   | >= 0.99   |
|:-----------------|------:|------:|------:|------:|------:|:------------------|:----------|
| Observada        | 0.85  | 0.858 | 0.884 | 1     | 1     | 0.0%              | 45.4%     |
| Regresión lineal | 0.9   | 0.908 | 0.917 | 0.926 | 0.933 | 0.0%              | 0.0%      |
| Red neuronal     | 0.898 | 0.907 | 0.916 | 0.923 | 0.931 | 0.0%              | 0.0%      |

*Fuente: elaboración propia con `python lgd_entrenar.py`. La red es la de mejor pérdida en validación entre las 30 semillas.*

![Figura 4](../resultados/lgd/distribucion.png)

**Figura 4.** Distribución de la LGD observada y de las predicciones de la regresión lineal y de la red en la prueba. *Fuente: elaboración propia.*

La LGD observada tiene dos grupos separados: el 45.4 % está en 1 y casi todo el resto está entre 0.80 y 0.95, con picos en 0.86 y 0.94. Las predicciones de los dos modelos, en cambio, se amontonan entre 0.90 y 0.93, cerca de la media (Tabla 12). Ninguno logra separar a quien no va a recuperar nada de quien va a recuperar el 14 %, así que ambos le predicen a casi todos algo intermedio que no le pasa a casi nadie. La regresión lineal tampoco se salió de [0, 1], porque sus predicciones quedaron en un rango muy estrecho; la ventaja teórica de la sigmoide no alcanzó a notarse aquí.

### 6.5 ¿Qué aportó la red en la LGD?

Nada que se pueda medir. Comparada con la regresión lineal:

- **Métricas:** igual o peor. MAE de 0.0763 contra 0.0756 y RMSE de 0.0937 contra 0.0927.
- **Variabilidad:** la red cambia con la semilla (desviación de 0.0006 en MAE); la regresión lineal da siempre lo mismo.
- **Costo:** 7105 parámetros contra 78, y 44 segundos para las 30 semillas más 46 de búsqueda, contra 0.1 segundos.
- **Interpretabilidad:** los coeficientes de la regresión se leen directamente; la red necesita métodos aparte.

La razón más probable es que la información que decide la LGD no está en la solicitud. Lo que se recupera depende de lo que pasa después del castigo, y la Tabla 8 sugiere que lo decide sobre todo el proceso de cobro. Una red más grande no puede aprender una relación que los datos de entrada no contienen. Para la pérdida esperada (sección 7), esto quiere decir que casi todo lo que distingue a un cliente de otro va a venir de la PD.

#### Aprendizajes de esta parte

- Antes de modelar hay que mirar cómo se construye la variable objetivo en el tiempo: los castigos recientes tienen LGD = 1 porque el cobro no ha terminado, no porque se pierda todo.
- Una distribución con picos tan marcados es una pista de que la variable la produce una regla del negocio, y que las características del cliente van a explicar poco.
- Elegir por validación puede dar una configuración que le va peor en la prueba. Cambiarla al ver la prueba invalida la prueba.
- Que la red no le gane a la media también es un resultado: muestra el límite de lo que se puede predecir con estos datos.

#### Referencias citadas en esta sección

Bergstra, J., y Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research, 13*(10), 281–305. https://www.jmlr.org/papers/v13/bergstra12a.html

Papke, L. E., y Wooldridge, J. M. (1996). Econometric methods for fractional response variables with an application to 401(k) plan participation rates. *Journal of Applied Econometrics, 11*(6), 619–632. https://doi.org/10.1002/(SICI)1099-1255(199611)11:6<619::AID-JAE418>3.0.CO;2-1

ranadeep. (2021). *Credit risk analysis* [Conjunto de datos]. Kaggle. https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset
