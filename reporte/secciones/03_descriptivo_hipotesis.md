<!-- Sección 3. Autor: Juan Pablo Pérez Álvarez.
     Rutas de figuras relativas a reporte/. Numeración de figuras y tablas provisional.
     Tablas: `python descriptivo.py` (resultados/descriptivo/hipotesis.md y verificacion.md). -->

## 3. Análisis descriptivo: ¿qué hace más riesgosa a una persona?

### 3.1 Hipótesis y cómo las probamos

Antes de mirar los datos escribimos diez hipótesis sobre qué hace más riesgosa a una persona. Cada una dice una variable y en qué dirección esperamos que se mueva el riesgo. Están escritas tal cual en `descriptivo.py`, en la lista `HIPOTESIS`.

Para probarlas comparamos la tasa de malos entre grupos, con los 268 530 créditos que tienen desenlace conocido:

- Si la variable es numérica, la cortamos en sus quintiles y comparamos el grupo de valores más altos con el de valores más bajos. Un mismo valor queda siempre en el mismo grupo. Por eso, en las variables con pocos valores distintos salen menos de cinco grupos: las consultas de crédito quedan en tres (0 a 1, 2, y 3 o más) y los años en el empleo, en cuatro. En esos casos la Tabla 5 nombra cada grupo por su rango.
- Si es categórica, comparamos los dos grupos que nombra la hipótesis; por ejemplo, quien arrienda contra quien tiene hipoteca.

Una hipótesis se confirma si la diferencia va en la dirección esperada y es de al menos un punto porcentual. El umbral tiene una razón. Un quintil tiene unos 53 700 créditos, y con una tasa cercana al 20 % el error estándar de la diferencia entre dos quintiles es de $\sqrt{2 \cdot 0.2 \cdot 0.8 / 53\,700} \approx 0.24$ puntos porcentuales. Un punto son entonces unos cuatro errores estándar, y una diferencia así difícilmente sale por azar. Los grupos de las variables con pocos valores son más grandes, así que allí el error es todavía menor.

Este análisis tiene dos límites. Mira una variable a la vez, así que una hipótesis puede confirmarse por culpa de otra variable relacionada; el plazo y el monto, por ejemplo, van de la mano. Además, muestra asociaciones, no causas. En la sección 5 volvemos a estas hipótesis con el modelo, que tiene en cuenta todas las variables al tiempo.

### 3.2 Resultados

La Tabla 5 resume las diez hipótesis y la Figura 2 muestra la tasa de malos en cada grupo.

**Tabla 5.** Hipótesis sobre el riesgo y tasa de malos en los grupos comparados.

| Hipótesis   | Qué esperábamos                                                               | Tasa de malos                          | ¿Se confirma?   |
|:------------|:------------------------------------------------------------------------------|:---------------------------------------|:----------------|
| H1          | Los créditos a 60 meses son más riesgosos que a 36                            | 60 meses: 34.6% vs 36 meses: 18.1%     | Sí              |
| H2          | Quien ya debe más en proporción a su ingreso es más riesgoso                  | quintil 5: 31.0% vs quintil 1: 15.6%   | Sí              |
| H3          | A mayor ingreso, menor riesgo                                                 | quintil 5: 16.8% vs quintil 1: 27.3%   | Sí              |
| H4          | Muchas consultas recientes de crédito indican necesidad de plata y más riesgo | 3 a 33: 26.7% vs 0 a 1: 20.9%          | Sí              |
| H5          | Quien usa casi todo su cupo de tarjetas es más riesgoso                       | quintil 5: 27.0% vs quintil 1: 15.3%   | Sí              |
| H6          | Una historia crediticia más larga indica menos riesgo                         | quintil 5: 20.8% vs quintil 1: 23.3%   | Sí              |
| H7          | Los préstamos más grandes son más riesgosos                                   | quintil 5: 26.6% vs quintil 1: 18.1%   | Sí              |
| H8          | Si el ingreso fue verificado, el riesgo es menor                              | Verified: 24.7% vs Not Verified: 17.1% | No              |
| H9          | Quien arrienda es más riesgoso que quien tiene casa o hipoteca                | RENT: 24.4% vs MORTGAGE: 19.6%         | Sí              |
| H10         | Más años en el mismo empleo, menos riesgo                                     | 9 a 10: 21.3% vs 0 a 2: 21.7%          | No              |

*Fuente: elaboración propia con datos de Lending Club (ranadeep, 2021).*

![Figura 2](../resultados/descriptivo/tasa_por_variable.png)

**Figura 2.** Tasa de malos por grupo para cada hipótesis. En las variables numéricas, los grupos son quintiles (1 = valores más bajos) o, si la variable tiene pocos valores distintos, rangos de valores. *Fuente: elaboración propia.*

Se confirman ocho de las diez. Las diferencias más grandes están en el plazo (H1: 34.6 % de malos a 60 meses contra 18.1 % a 36) y en la deuda frente al ingreso (H2: 31.0 % en el quintil más alto de `dti` contra 15.6 % en el más bajo). Les siguen el uso del cupo de las tarjetas (H5), el ingreso (H3), el monto del préstamo (H7) y las consultas recientes de crédito (H4: 26.7 % con tres o más consultas en seis meses, 20.9 % con una o ninguna). En la Figura 2, la tasa de estas variables sube o baja paso a paso de un grupo al siguiente, y no solo entre los extremos.

H6 se confirma, pero por poco: la diferencia es de 2.5 puntos (20.8 % contra 23.3 %). H9 también se confirma: quien arrienda tiene 24.4 % de malos y quien tiene hipoteca, 19.6 %.

H10 no se confirma. La tasa de malos casi no cambia con la antigüedad en el empleo: 21.7 % con dos años o menos y 21.3 % con nueve o diez, y los grupos del medio no siguen ningún orden. Saber cuántos años lleva alguien en su trabajo no ayuda, por sí solo, a separar buenos de malos.

Una primera versión de esta tabla repartía los valores repetidos entre quintiles vecinos para que todos quedaran del mismo tamaño. El reparto seguía el orden del archivo, que va por fecha, y por eso en H4 dos grupos con el mismo valor (cero consultas) salían con tasas distintas: 18.3 % y 20.1 %. Al corregirlo cambiaron las cifras de H4 y H10, pero no las conclusiones. El hallazgo A2 de la sección 10 cuenta cómo lo encontramos.

### 3.3 La hipótesis que salió al revés: el ingreso verificado

H8 salió al contrario de lo esperado. Los créditos con ingreso verificado tienen más malos (24.7 %) que los no verificados (17.1 %). Nuestra primera sospecha fue el monto: si Lending Club verifica más a quien pide más plata, y los préstamos grandes son más riesgosos (H7), la verificación estaría heredando ese riesgo. La Tabla 6 lo revisa por quintil de monto.

**Tabla 6.** Proporción de créditos sin verificar y tasa de malos por estado de verificación, dentro de cada quintil del monto del préstamo.

|   Quintil de monto | Monto máximo   | Sin verificar   | Malos: Not Verified   | Malos: Source Verified   | Malos: Verified   |
|-------------------:|:---------------|:----------------|:----------------------|:-------------------------|:------------------|
|                  1 | 6 250          | 51.9%           | 16.5%                 | 18.5%                    | 21.3%             |
|                  2 | 10 000         | 44.9%           | 16.9%                 | 21.6%                    | 22.6%             |
|                  3 | 14 400         | 37.6%           | 16.8%                 | 25.4%                    | 24.4%             |
|                  4 | 20 000         | 26.2%           | 18.2%                 | 25.9%                    | 25.6%             |
|                  5 | 35 000         | 5.1%            | 21.2%                 | 27.4%                    | 26.7%             |

*Fuente: elaboración propia con datos de Lending Club (ranadeep, 2021). Montos en dólares.*

La verificación depende mucho del monto: en el quintil más bajo la mitad de los créditos (51.9 %) no se verificó, y en el más alto solo el 5.1 %. Pero el monto no lo explica todo, porque dentro de cada quintil los verificados siguen teniendo más malos: 21.3 % contra 16.5 % en el primero y 26.7 % contra 21.2 % en el quinto.

Falta otra variable: el tiempo. En 2008 el 85 % de los créditos no se verificaba, y en 2015 solo el 26 %. En esos mismos años sube la tasa de malos, en parte por el sesgo de madurez de la sección 2. Comparando solo créditos del mismo año y plazo (Tabla P13, en la sección 5), la brecha baja de 7.6 a 2.9 puntos. De 2010 a 2013 queda entre −0.3 y +1.8 puntos, salvo a 60 meses en 2012, y solo es grande en 2014 y 2015, justo los años con sesgo de madurez. Casi toda la diferencia de H8 sale entonces del monto y del año en que se originó el crédito. El modelo de la sección 5 apunta a lo mismo. Cuando tiene en cuenta todas las variables a la vez, pasar de no verificado a verificado sube la probabilidad de no pagar apenas 0.1 puntos porcentuales en la red (Tabla P12).

Queda una brecha pequeña que no sabemos explicar. Puede venir de cómo decide Lending Club a quién le pide la verificación, pero no encontramos una fuente primaria que describa ese criterio, así que lo dejamos como pregunta abierta. Lo que sí está claro es que verificar o no el ingreso no vuelve a nadie mejor o peor pagador. Por eso, en la aplicación, el "¿qué pasaría si...?" no ofrece cambiar el estado de verificación.

#### Referencias citadas en esta sección

ranadeep. (2021). *Credit risk analysis* [Conjunto de datos]. Kaggle. https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset
