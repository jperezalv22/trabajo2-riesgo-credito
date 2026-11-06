<!-- Sección 1. Autores: Sebastián (parte C), con Juan Pablo. Borrador para que Juan Pablo lo revise.
     Los enlaces {{...}} los llena construir_blog.py desde config.yaml (app.enlaces). -->

## 1. Introducción y caso de uso

Antes de prestarle dinero a alguien, un banco se hace dos preguntas: qué tan probable es que no le paguen y, si no le pagan, cuánto va a perder. La primera es la **probabilidad de incumplimiento** (PD, por sus siglas en inglés), un problema de clasificación binaria. La segunda es la **pérdida dado incumplimiento** (LGD), la fracción de lo expuesto que finalmente se pierde, un problema de regresión. Juntas, con la **exposición al incumplimiento** (EAD), dan la pérdida esperada de cada crédito:

$$
\mathbb{E}[L] = \text{PD} \cdot \text{LGD} \cdot \text{EAD}
$$

Es la misma descomposición del acuerdo de Basilea II, que define la pérdida esperada de una exposición como $\text{PD} \times \text{LGD}$ y su monto como esa pérdida esperada multiplicada por la EAD (Basel Committee on Banking Supervision [BCBS], 2006, párrs. 375-376).

En este trabajo respondemos las dos preguntas con perceptrones multicapa y las comparamos, bajo las mismas condiciones, con un modelo simple: una regresión logística para la PD y una regresión lineal para la LGD. Las redes neuronales no siempre son la mejor herramienta para datos tabulares, así que el objetivo no es mostrar que la red gana, sino averiguar con evidencia qué aporta y qué no. Además, convertimos el modelo de PD en una scorecard y en una aplicación web en la que cualquier persona puede ver su puntaje.

### 1.1 Caso de uso

Todas las decisiones de modelado del reporte se desprenden de esta descripción:

- **Quién lo usa.** El área de originación de una entidad que presta créditos de consumo por internet, como Lending Club, cuyos datos usamos (sección 2). Una versión simplificada la usa también la persona que va a pedir el crédito, en la aplicación web: le muestra cómo la vería el banco antes de presentar la solicitud.
- **Para qué decisión.** Aprobar o rechazar la solicitud y estimar cuánto se espera perder con ella. La PD y la scorecard ordenan a los solicitantes y permiten fijar un punto de corte; la LGD y la pérdida esperada sirven para provisionar y para comparar el riesgo de la cartera.
- **En qué momento.** Cuando llega la solicitud, antes de aprobar el crédito.
- **Con qué información.** Solo con la que existe en ese momento: lo que la persona declara en la solicitud (ingreso, monto, plazo, propósito, vivienda, antigüedad en el empleo) y lo que dice su reporte de crédito (moras, consultas, líneas abiertas, uso del cupo, registros públicos).

### 1.2 Lo que esto implica para el modelado

1. **Nada del futuro.** No usamos variables de pagos (`total_pymnt`, `recoveries`, `last_pymnt_d`...), porque el día de la solicitud no existen. Tampoco la calificación (`grade`, `sub_grade`) ni la tasa de interés o la cuota que asigna Lending Club, porque son el resultado de su propio modelo de riesgo: usarlas sería copiar ese modelo en lugar de construir el nuestro (sección 2.3).
2. **Partición temporal.** El modelo se entrena con el pasado y se usa con clientes nuevos. Por eso la prueba son los créditos originados después de los de entrenamiento: 2014-2015 en la PD y 2013 en la LGD (secciones 2.4 y 6.1).
3. **La probabilidad tiene que estar bien calibrada,** no solo ordenar bien, porque la PD entra multiplicando en la pérdida esperada. Por eso medimos la calibración (Brier y curva de calibración) y no balanceamos las clases (sección 5.1).
4. **El resultado se tiene que poder explicar.** Al solicitante y al analista no les sirve solo un número: la scorecard reparte el puntaje entre las características de la persona (sección 5.5), y la aplicación lo muestra.
5. **La LGD también se estima al originar,** con las mismas variables de la solicitud, aunque se mida sobre los créditos que ya se castigaron (sección 6).

### 1.3 Contenido del reporte

La sección 2 describe los datos y la variable objetivo; la 3, el análisis descriptivo y las hipótesis sobre qué hace más riesgosa a una persona; la 4, la fundamentación matemática de las redes, las pérdidas, el entrenamiento y las métricas. Las secciones 5 y 6 presentan los modelos de PD (con la scorecard y la interpretación de las variables) y de LGD, y la 7 los combina en la pérdida esperada. La 8 discute qué aportó la red frente a los modelos de referencia y qué aprendimos. Las secciones 9 y 10 cuentan cómo usamos la inteligencia artificial y qué errores suyos encontramos.

### 1.4 Enlaces

- Repositorio con el código, los datos procesados y los modelos: {{repositorio}}
- Aplicación web para conocer su puntaje: {{app}}
- Video promocional: {{video}}

#### Referencias citadas en esta sección

Basel Committee on Banking Supervision. (2006). *International convergence of capital measurement and capital standards: A revised framework. Comprehensive version*. Bank for International Settlements. https://www.bis.org/publ/bcbs128.htm
