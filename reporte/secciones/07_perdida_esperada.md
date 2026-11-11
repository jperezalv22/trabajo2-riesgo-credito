<!-- Sección 7. Autor: Juan Pablo Pérez Álvarez.
     Rutas de figuras relativas a reporte/. Numeración de figuras y tablas provisional.
     Tablas y figura: `python perdida_esperada.py` (resultados/perdida_esperada/), 12 s. -->

## 7. Pérdida esperada

La pérdida esperada de un crédito junta las dos partes del trabajo:

$$
E[L]_i = \text{PD}_i \cdot \text{LGD}_i \cdot \text{EAD}_i
$$

Es la descomposición de Basilea II que presentamos en la sección 1 (BCBS, 2006). La PD sale de la red de la sección 5, la misma que usa la aplicación, y la LGD de la red de la sección 6. Ninguna se reentrena: `perdida_esperada.py` carga los modelos guardados y calcula la pérdida esperada de los 268 530 créditos con desenlace conocido en 12 segundos.

Nuestra PD es la probabilidad de que el crédito termine mal en algún momento de su vida, no en los próximos doce meses como en Basilea. Por eso $E[L]$ es la pérdida esperada durante toda la vida del crédito. Tampoco descontamos el valor del dinero en el tiempo, igual que en la LGD.

### 7.1 La exposición al incumplimiento (EAD)

La EAD es cuánto dinero está en riesgo si la persona deja de pagar. La opción más simple es el monto desembolsado (`funded_amnt`), que es lo que el banco expone el primer día. Pero los créditos de Lending Club se pagan en cuotas fijas, y cada cuota abona a capital, así que quien deja de pagar casi nunca debe todo el monto. En la sección 6 medimos la EAD de cada castigado como el capital que le faltaba por pagar. Esa EAD no sirve aquí, porque solo se conoce después del castigo y la pérdida esperada se calcula al aprobar el crédito.

Usamos entonces el monto desembolsado por la fracción de capital que suele seguir pendiente cuando un crédito se castiga:

$$
\text{EAD}_i = \texttt{funded\_amnt}_i \cdot f(\text{plazo}_i), \qquad f(p) = \frac{\sum_{j \,:\, \text{plazo}_j = p} \text{EAD}_j}{\sum_{j \,:\, \text{plazo}_j = p} \texttt{funded\_amnt}_j}
$$

donde las sumas recorren los créditos castigados de entrenamiento de la LGD (originados hasta 2012). La Tabla 13 muestra el factor.

**Tabla 13.** Fracción del monto desembolsado que seguía pendiente al castigo, por plazo, en los castigados originados hasta 2012.

| Plazo    | Castigados   | Capital pendiente al castigo / monto   |
|:---------|:-------------|:---------------------------------------|
| 36 meses | 9 720        | 60.1%                                  |
| 60 meses | 4 699        | 76.0%                                  |

*Fuente: elaboración propia con datos de Lending Club (ranadeep, 2021).*

Lo calculamos por plazo porque un crédito a 60 meses abona menos capital en cada cuota y, cuando cae, debe una parte mayor: 76.0 % del monto contra 60.1 % a 36 meses. Usamos la razón de las sumas, y no el promedio de las razones de cada crédito, porque la pérdida esperada se suma en dólares y un crédito grande pesa más. Las dos cosas que necesita el factor, el monto y el plazo, se conocen al pedir el crédito, y el factor sale solo de los años de entrenamiento.

Tiene una limitación clara: es un promedio. Quien deja de pagar en el tercer mes debe casi todo, y quien deja de pagar en el mes 30 debe poco. No modelamos cuándo cae cada persona.

### 7.2 Resultados

La Tabla 14 compara la pérdida esperada con la observada en tres grupos de años de originación. La pérdida observada es el capital que se perdió de verdad: la EAD por la LGD de cada castigado, y cero en los créditos que se pagaron.

**Tabla 14.** Pérdida esperada y observada como porcentaje del monto desembolsado, por año de originación.

| Originación   | Créditos   | Monto (millones de USD)   | PD media   | Tasa de malos   | LGD media predicha   | EAD / monto   | Pérdida esperada   | Pérdida esperada con EAD = monto   | Pérdida observada   | Malos sin pérdida conocida   |
|:--------------|:-----------|:--------------------------|:-----------|:----------------|:---------------------|:--------------|:-------------------|:-----------------------------------|:--------------------|:-----------------------------|
| 2007 a 2012   | 90 347     | 1 073                     | 18.1%      | 16.3%           | 0.915                | 64.1%         | 12.0%              | 18.0%                              | 10.8%               | 2.1%                         |
| 2013          | 73 194     | 1 057                     | 20.3%      | 22.9%           | 0.912                | 65.2%         | 13.6%              | 20.0%                              | 14.3%               | 11.7%                        |
| 2014 y 2015   | 104 989    | 1 514                     | 21.8%      | 26.0%           | 0.912                | 66.4%         | 14.9%              | 21.5%                              | 13.7%               | 38.6%                        |

*Fuente: elaboración propia con datos de Lending Club (ranadeep, 2021). "Malos sin pérdida conocida" son los créditos en Default o Late (31-120 days): todavía no se castigan, así que su pérdida observada cuenta como cero.*

Lo primero que muestra la tabla es que la EAD pesa más que los modelos. Con la EAD igual al monto completo, la pérdida esperada queda entre 18.0 % y 21.5 % de lo prestado, de 1.4 a 1.7 veces la observada. Con el factor de la Tabla 13 baja a entre 12.0 % y 14.9 %, del mismo orden que lo observado.

La comparación no es una prueba limpia en ningún grupo, y conviene decir por qué en cada uno:

- **2007 a 2012.** Casi todas las pérdidas ya se conocen (solo falta el 2.1 % de los malos), pero son años de entrenamiento para las dos redes. La pérdida esperada (12.0 %) pasa a la observada (10.8 %) porque la PD también se pasa: 18.1 % contra 16.3 % de malos. La red aprendió de todos los años hasta 2013 a la vez, y los primeros tuvieron menos malos que el promedio.
- **2013.** Es la prueba de la LGD, pero la PD aprendió de estos créditos. La pérdida esperada (13.6 %) queda 0.7 puntos por debajo de la observada (14.3 %), y la observada todavía va a subir, porque al 11.7 % de los malos aún no se le conoce la pérdida.
- **2014 y 2015.** Es la prueba de la PD, pero la pérdida observada está incompleta en dos sentidos. El 38.6 % de los malos todavía no tiene pérdida conocida, lo que la baja. Y los castigados que sí se conocen son los que cayeron rápido, con más capital pendiente: su EAD es entre el 81 % y el 92 % del monto según el decil de la Tabla 16, y el factor supone entre el 60 % y el 76 %. Eso la sube. El total observado (13.7 %) sale de esos dos errores compensándose, y no dice si los modelos aciertan.

La Tabla 15 cambia los modelos en 2013, el grupo donde la comparación es menos mala. Las cuatro combinaciones quedan a menos de 0.15 puntos entre sí. En el total de la cartera da igual usar las redes o los modelos simples, cosa que ya anticipaban las secciones 5 y 6. En 2013, la decisión sobre la EAD mueve el resultado 6.4 puntos, y la de los modelos, menos de dos décimas.

**Tabla 15.** Pérdida esperada de los créditos originados en 2013 con cada combinación de modelos (EAD con el factor de la Tabla 13).

| PD        | LGD              | Pérdida esperada   | Pérdida observada   | Esperada - observada (puntos %)   |
|:----------|:-----------------|:-------------------|:--------------------|:----------------------------------|
| Red       | Red              | 13.63%             | 14.35%              | -0.71                             |
| Red       | Regresión lineal | 13.71%             | 14.35%              | -0.64                             |
| Logística | Red              | 13.56%             | 14.35%              | -0.79                             |
| Logística | Regresión lineal | 13.64%             | 14.35%              | -0.71                             |

*Fuente: elaboración propia.*

Para un banco, la pregunta útil es si la pérdida esperada ordena bien los créditos: si los que el modelo marca como caros de verdad pierden más. La Figura 5 y la Tabla 16 parten los créditos en diez grupos según su pérdida esperada por dólar prestado.

![Figura 5](../resultados/perdida_esperada/deciles.png)

**Figura 5.** Pérdida esperada y observada por decil de pérdida esperada, como porcentaje del monto desembolsado. *Fuente: elaboración propia.*

**Tabla 16.** Pérdida esperada, pérdida observada y tasa de malos por decil de pérdida esperada (1 = menor).

| Decil   | Esperada (2013)   | Observada (2013)   | Tasa de malos (2013)   | Esperada (2014 y 2015)   | Observada (2014 y 2015)   | Tasa de malos (2014 y 2015)   |
|:--------|:------------------|:-------------------|:-----------------------|:-------------------------|:--------------------------|:------------------------------|
| 1       | 2.8%              | 2.9%               | 5.6%                   | 2.9%                     | 3.7%                      | 7.5%                          |
| 2       | 4.9%              | 4.8%               | 10.0%                  | 5.1%                     | 5.9%                      | 12.5%                         |
| 3       | 6.3%              | 6.2%               | 12.4%                  | 6.8%                     | 7.6%                      | 15.7%                         |
| 4       | 7.7%              | 7.8%               | 16.1%                  | 8.3%                     | 9.5%                      | 19.8%                         |
| 5       | 9.1%              | 10.7%              | 20.3%                  | 9.9%                     | 11.3%                     | 24.0%                         |
| 6       | 10.6%             | 12.2%              | 23.0%                  | 11.8%                    | 12.8%                     | 27.3%                         |
| 7       | 12.6%             | 14.3%              | 27.5%                  | 14.3%                    | 14.9%                     | 31.7%                         |
| 8       | 15.7%             | 16.9%              | 31.0%                  | 18.1%                    | 16.1%                     | 34.0%                         |
| 9       | 21.4%             | 22.0%              | 35.1%                  | 24.0%                    | 18.9%                     | 37.9%                         |
| 10      | 31.9%             | 32.4%              | 48.6%                  | 33.1%                    | 26.2%                     | 49.5%                         |

*Fuente: elaboración propia.*

En 2013 el orden se cumple de punta a punta. El decil más barato pierde el 2.9 % de lo prestado y el más caro el 32.4 %, once veces más, y en cada decil la pérdida observada queda a 1.7 puntos o menos de la esperada. En 2014 y 2015 el orden también se cumple, pero los dos extremos se separan: en los deciles bajos se pierde más de lo esperado y en los altos, menos. Los deciles altos son casi todos créditos a 60 meses (el 99 % del décimo), y para ellos pesa más el problema de la madurez que describimos arriba.

La Tabla 17 muestra tres créditos de 2014 y 2015 para ver la fórmula con números.

**Tabla 17.** Tres créditos originados en 2014 y 2015, en los percentiles 10, 50 y 90 de la pérdida esperada por dólar prestado.

| Percentil   | Monto   | Plazo    | PD    | LGD   | EAD    | E[L] (USD)   | E[L] / monto   | Desenlace   |
|:------------|:--------|:---------|:------|:------|:-------|:-------------|:---------------|:------------|
| 10%         | 3 600   | 36 meses | 7.6%  | 0.918 | 2 164  | 152          | 4.2%           | bueno       |
| 50%         | 7 200   | 36 meses | 19.6% | 0.913 | 4 328  | 774          | 10.8%          | malo        |
| 90%         | 18 000  | 60 meses | 39.6% | 0.918 | 13 681 | 4 974        | 27.6%          | bueno       |

*Fuente: elaboración propia. Montos en dólares.*

El crédito del medio es de 7 200 dólares a 36 meses. La red le da una PD de 19.6 % y una LGD de 0.913, y su EAD es 7 200 × 60.1 % = 4 328 dólares. La pérdida esperada es 0.196 × 0.913 × 4 328 ≈ 774 dólares, el 10.8 % de lo prestado. Para no perder plata con créditos como ese, los intereses tendrían que cubrir al menos esos 774 dólares, más el costo del dinero y la operación. Este crédito terminó mal y el del percentil 90 terminó bien. La pérdida esperada es un promedio sobre muchos créditos parecidos y no dice cuál de ellos va a caer.

### 7.3 Limitaciones

- La LGD casi no distingue entre créditos. En el 90 % de los créditos la red predice entre 0.89 y 0.94, mientras la PD va de 5 % a 45 %. Las diferencias en la pérdida esperada salen casi todas de la PD y de la EAD.
- La PD se queda corta en los años recientes: en 2014 y 2015 predice 21.8 % de malos y hubo 26.0 %. Parte de esa brecha es sesgo de madurez y no riesgo real (sección 5), pero no sabemos cuánta, así que tampoco sabemos cuánto subestima la pérdida esperada en esos años.
- El factor de exposición es un promedio de los castigados hasta 2012. Si la gente empieza a caer antes, como se ve en los datos de 2014 y 2015, la EAD real es mayor.
- Solo contamos capital. No entran los intereses que se dejan de cobrar, los costos internos de cobranza ni el valor del dinero en el tiempo.

#### Referencias citadas en esta sección

Basel Committee on Banking Supervision. (2006). *International convergence of capital measurement and capital standards: A revised framework. Comprehensive version*. Bank for International Settlements. https://www.bis.org/publ/bcbs128.htm

ranadeep. (2021). *Credit risk analysis* [Conjunto de datos]. Kaggle. https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset
