<!-- Sección 3. Autor: Juan Pablo Pérez Álvarez.
     Rutas de figuras relativas a reporte/. Numeración de figuras y tablas provisional.
     Tablas: `python descriptivo.py` (resultados/descriptivo/hipotesis.md y verificacion.md). -->

## 3. Análisis descriptivo: ¿qué hace más riesgosa a una persona?

### 3.1 Hipótesis y cómo las probamos

Antes de mirar los datos escribimos diez hipótesis sobre qué hace más riesgosa a una persona. Cada una dice una variable y en qué dirección esperamos que se mueva el riesgo. Están escritas tal cual en `descriptivo.py`, en la lista `HIPOTESIS`.

Para probarlas comparamos la tasa de malos entre grupos, con los 268 530 créditos que tienen desenlace conocido:

- Si la variable es numérica, partimos a las personas en quintiles (cinco grupos del mismo tamaño) y comparamos el quintil más alto con el más bajo.
- Si es categórica, comparamos los dos grupos que nombra la hipótesis; por ejemplo, quien arrienda contra quien tiene hipoteca.

Una hipótesis se confirma si la diferencia va en la dirección esperada y es de al menos un punto porcentual. Ese umbral no es arbitrario del todo. Cada quintil tiene unos 53 700 créditos, y con una tasa cercana al 20 % el error estándar de la diferencia entre dos quintiles es de $\sqrt{2 \cdot 0.2 \cdot 0.8 / 53\,700} \approx 0.24$ puntos porcentuales. Un punto equivale entonces a unos cuatro errores estándar: una diferencia de ese tamaño difícilmente sale por azar.

Este análisis tiene dos límites. Mira una variable a la vez, así que una hipótesis puede confirmarse por culpa de otra variable relacionada; el plazo y el monto, por ejemplo, van de la mano. Además, muestra asociaciones, no causas. En la sección 5 volvemos a estas hipótesis con el modelo, que sí tiene en cuenta todas las variables al tiempo.

### 3.2 Resultados

La Tabla 5 resume las diez hipótesis y la Figura 2 muestra la tasa de malos en cada grupo.

**Tabla 5.** Hipótesis sobre el riesgo y tasa de malos en los grupos comparados.

| Hipótesis   | Qué esperábamos                                                               | Tasa de malos                          | ¿Se confirma?   |
|:------------|:------------------------------------------------------------------------------|:---------------------------------------|:----------------|
| H1          | Los créditos a 60 meses son más riesgosos que a 36                            | 60 meses: 34.6% vs 36 meses: 18.1%     | Sí              |
| H2          | Quien ya debe más en proporción a su ingreso es más riesgoso                  | quintil 5: 31.0% vs quintil 1: 15.6%   | Sí              |
| H3          | A mayor ingreso, menor riesgo                                                 | quintil 5: 16.8% vs quintil 1: 27.3%   | Sí              |
| H4          | Muchas consultas recientes de crédito indican necesidad de plata y más riesgo | quintil 5: 26.0% vs quintil 1: 18.3%   | Sí              |
| H5          | Quien usa casi todo su cupo de tarjetas es más riesgoso                       | quintil 5: 27.0% vs quintil 1: 15.3%   | Sí              |
| H6          | Una historia crediticia más larga indica menos riesgo                         | quintil 5: 20.8% vs quintil 1: 23.3%   | Sí              |
| H7          | Los préstamos más grandes son más riesgosos                                   | quintil 5: 26.5% vs quintil 1: 18.1%   | Sí              |
| H8          | Si el ingreso fue verificado, el riesgo es menor                              | Verified: 24.7% vs Not Verified: 17.1% | No              |
| H9          | Quien arrienda es más riesgoso que quien tiene casa o hipoteca                | RENT: 24.4% vs MORTGAGE: 19.6%         | Sí              |
| H10         | Más años en el mismo empleo, menos riesgo                                     | quintil 5: 21.3% vs quintil 1: 21.5%   | No              |

*Fuente: elaboración propia con datos de Lending Club (ranadeep, 2021).*

![Figura 2](../resultados/descriptivo/tasa_por_variable.png)

**Figura 2.** Tasa de malos por grupo para cada hipótesis. En las variables numéricas, los grupos son quintiles (1 = valores más bajos). *Fuente: elaboración propia.*

Se confirman ocho de las diez. Las diferencias más grandes están en el plazo (H1: 34.6 % de malos a 60 meses contra 18.1 % a 36) y en la deuda frente al ingreso (H2: 31.0 % en el quintil más alto de `dti` contra 15.6 % en el más bajo). Les siguen el uso del cupo de las tarjetas (H5), el ingreso (H3), el monto del préstamo (H7) y las consultas recientes de crédito (H4). En la Figura 2 se ve que en todas estas la tasa sube o baja de forma ordenada de un quintil al siguiente, no solo entre los extremos.

H6 se confirma, pero por poco: la diferencia es de 2.5 puntos (20.8 % contra 23.3 %). H9 también se confirma: quien arrienda tiene 24.4 % de malos y quien tiene hipoteca, 19.6 %.

H10 no se confirma. La tasa de malos es prácticamente la misma en todos los quintiles de antigüedad en el empleo (21.3 % en el más alto y 21.5 % en el más bajo). Saber cuántos años lleva alguien en su trabajo no ayuda, por sí solo, a separar buenos de malos.

### 3.3 La hipótesis que salió al revés: el ingreso verificado

H8 salió al contrario de lo esperado. Los créditos con ingreso verificado tienen más malos (24.7 %) que los no verificados (17.1 %). Nuestra primera sospecha fue el monto: si Lending Club verifica más a quien pide más plata, y los préstamos grandes son más riesgosos (H7), la verificación estaría heredando ese riesgo. La Tabla 6 lo revisa por quintil de monto.

**Tabla 6.** Proporción de créditos sin verificar y tasa de malos por estado de verificación, dentro de cada quintil del monto del préstamo.

|   Quintil de monto | Monto máximo   | Sin verificar   | Malos: Not Verified   | Malos: Source Verified   | Malos: Verified   |
|-------------------:|:---------------|:----------------|:----------------------|:-------------------------|:------------------|
|                  1 | 6 250          | 51.9%           | 16.5%                 | 18.5%                    | 21.3%             |
|                  2 | 10 000         | 45.6%           | 16.8%                 | 20.7%                    | 22.3%             |
|                  3 | 14 400         | 38.3%           | 17.0%                 | 25.3%                    | 24.4%             |
|                  4 | 20 000         | 27.7%           | 18.1%                 | 26.2%                    | 25.6%             |
|                  5 | 35 000         | 5.3%            | 21.4%                 | 27.0%                    | 26.7%             |

*Fuente: elaboración propia con datos de Lending Club (ranadeep, 2021). Montos en dólares.*

Como muestra la Tabla 6, la sospecha era en parte cierta. La verificación depende mucho del monto: en el quintil más bajo la mitad de los créditos (51.9 %) no se verificó, y en el más alto solo el 5.3 %. Pero eso no explica todo, porque dentro de cada quintil los verificados siguen siendo más riesgosos que los no verificados: 21.3 % contra 16.5 % en el primero y 26.7 % contra 21.4 % en el quinto.

Concluimos que la verificación no se asigna al azar. Lending Club decide a quién se la pide, y si se la pide más a quien le parece más riesgoso por razones que no vemos en los datos, el grupo verificado carga con ese riesgo. Esto último es una conjetura: no encontramos una fuente primaria de Lending Club que describa su criterio de verificación, así que lo dejamos como pregunta abierta.

Para un modelo predictivo, "verificado" sí indica más riesgo y es correcto usarlo. Como consejo sería un error: no verificar el ingreso no hace a nadie mejor pagador. Por eso, en la aplicación, el "¿qué pasaría si...?" no debería sugerir cambiar el estado de verificación como forma de mejorar el puntaje.

#### Referencias citadas en esta sección

ranadeep. (2021). *Credit risk analysis* [Conjunto de datos]. Kaggle. https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset
