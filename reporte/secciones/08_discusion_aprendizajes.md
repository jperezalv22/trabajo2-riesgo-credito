<!-- Sección 8. Autor: Sebastián (parte C), con todo el equipo. Borrador: Juan Pablo y Santiago deben
     revisar que las conclusiones de sus partes queden bien dichas y agregar sus aprendizajes.
     FALTA: la parte de pérdida esperada (8.1 y 8.4) cuando exista la sección 7.
     Numeración provisional: Tabla D1. Cifras de resultados/pd/ y resultados/lgd/. -->

## 8. Discusión y aprendizajes

### 8.1 ¿Qué aportó la red neuronal?

La Tabla D1 pone lado a lado las dos tareas con los criterios que pide la regla de comparación justa: las métricas y su variabilidad, el presupuesto de búsqueda, el costo de cómputo y la complejidad. Los dos modelos de cada tarea usan los mismos datos, las mismas particiones y el mismo preprocesamiento.

**Tabla D1.** La red neuronal frente al modelo de referencia en cada tarea, en la prueba temporal.

| Tarea | Métrica | Modelo de referencia | Red neuronal (30 semillas) | Parámetros (referencia / red) | Configuraciones probadas (referencia / red) | Tiempo total (referencia / red) |
|:--|:--|:--|:--|:--|:--|:--|
| PD | AUC | 0.6797 | 0.6878 ± 0.0012 | 90 / 19 841 | 5 / 30 | 2.1 s / 762 s |
| PD | Brier | 0.1795 | 0.1781 ± 0.0003 | | | |
| LGD | MAE | 0.0756 | 0.0763 ± 0.0006 | 78 / 7 105 | 1 / 12 | 0.1 s / 90 s |
| LGD | RMSE | 0.0927 | 0.0937 ± 0.0011 | | | |

*Fuente: elaboración propia a partir de las Tablas P3, P15 y 11. El modelo de referencia es la regresión logística en la PD y la regresión lineal en la LGD. El tiempo de la red incluye la búsqueda de hiperparámetros y las 30 semillas.*

**En la PD, poco, pero de manera consistente.** La red mejora el AUC en 0.008 y el Brier en 0.0015. La ventaja no es suerte de una semilla: gana en las 30 (Tabla P4) y el intervalo bootstrap de la diferencia de AUC no incluye el cero (sección 5.3). Pero no mejora la calibración fuera de tiempo, cuesta unas 360 veces más tiempo y tiene 220 veces más parámetros, y tuvo seis veces más intentos de búsqueda que la logística. Agregar las cuatro variables de Lending Club a la logística, que es agregar información y no complejidad, sube el AUC de la prueba a 0.6947, más que la red (Tabla P14).

**En la LGD, nada medible.** La red no le gana a la regresión lineal en ninguna métrica, y en RMSE ninguno de los dos le gana a predecir siempre la media (Tabla 11). La sección 6.5 explica la razón más probable: lo que se recupera depende del proceso de cobro, que ocurre después del castigo, y no de lo que se sabe en la solicitud.

**Interpretabilidad.** La regresión logística tiene una tabla de puntos fija: cada variable suma o resta lo mismo para todos. La red necesita un método aparte, los valores de Shapley, para repartir el puntaje de cada persona (sección 5.5). Los dos modelos coinciden en las variables que más pesan: el plazo, el ingreso, el uso del cupo rotativo y el monto (sección 5.6).

Esto coincide con lo que reportan estudios más amplios. En 45 conjuntos de datos tabulares de tamaño medio, los modelos de árboles siguen superando a las redes (Grinsztajn et al., 2022), y XGBoost supera a varias redes diseñadas para datos tabulares con mucho menos ajuste de hiperparámetros (Shwartz-Ziv y Armon, 2022). Nuestros datos son más grandes que los de esos estudios y nosotros no probamos árboles, así que no podemos decir si un árbol habría ganado aquí. Lo que sí muestran nuestros resultados es que, con estas variables, lo que la red puede agregar en no linealidades e interacciones vale menos de una centésima de AUC: un modelo lineal en el logaritmo de las odds ya captura casi todo.

¿Cuál dejaríamos en producción? En la aplicación dejamos la red, porque es la que mejor ordena y sus puntos cuadran con el puntaje. Pero un banco que tenga que explicarle a cada cliente por qué lo rechazó, con una tabla fija y auditable, tendría buenas razones para preferir la logística y renunciar a 0.008 de AUC. Con `--set scorecard.modelo=logistica` la aplicación usa la logística sin cambiar el código.

### 8.2 Las hipótesis, al final

De las diez hipótesis de la sección 3, los datos por sí solos confirmaron ocho (Tabla 5). Al controlar las demás variables con los modelos, quedan siete (Tabla P12):

- **Se confirman** H1 (plazo), H2 (deuda sobre ingreso), H3 (ingreso), H4 (consultas recientes), H5 (uso del cupo), H7 (monto) y H9 (arrendar).
- **H6 (historia crediticia más larga, menos riesgo)** se confirma en los datos, pero desaparece en los modelos. Una explicación posible, que no comprobamos: la antigüedad de la historia está correlacionada con el número total de cuentas (correlación de Spearman de 0.41 en los créditos hasta 2013) y con el ingreso (0.31), y en los modelos esas variables ya cuentan lo que ella decía.
- **H8 (verificación del ingreso)** sale al revés en los datos y casi en cero en los modelos. La brecha viene de a quién decidía verificar Lending Club y de cómo cambió eso con los años (secciones 3.3 y 5.6). Por eso la aplicación no ofrece cambiarla en el "¿qué pasaría si...?".
- **H10 (años en el empleo)** no se confirma ni en los datos ni en los modelos.

### 8.3 Limitaciones

- **Solo créditos aprobados.** El conjunto solo trae créditos que Lending Club otorgó. El diccionario describe también un archivo de solicitudes rechazadas (hoja *RejectStats File*), que no viene en este conjunto. El modelo aprende el riesgo entre quienes ya pasaron el filtro de Lending Club, y en la aplicación lo usamos con cualquier persona. En alguien muy distinto a los aprobados, la PD es una extrapolación.
- **Sesgo de madurez.** En la prueba de la PD solo tienen desenlace los créditos que terminaron rápido. Eso explica, al menos en parte, por qué la PD se queda corta en 2014-2015 (sección 5.4), y no lo podemos separar de un cambio real en el riesgo.
- **LGD sin el valor del dinero en el tiempo.** No hay fechas de recuperación, así que no podemos descontar lo recuperado; la LGD real sería algo mayor (sección 6.1).
- **Datos de Estados Unidos entre 2007 y 2015.** Las relaciones que encontramos no tienen por qué valer para otro país, otro producto u otra época.

### 8.4 Aprendizajes

Sobre el problema:

1. **La información pesa más que la arquitectura.** En la PD, la logística con las variables de Lending Club le gana a la red sin ellas; en la LGD, ninguna arquitectura puede aprender lo que la solicitud no contiene.
2. **Ordenar bien y acertar la probabilidad son cosas distintas.** El modelo sigue ordenando a los clientes en 2014-2015 aunque sus PD se queden cortas. Por eso la discriminación y la calibración se miden por separado.
3. **Cómo se construye la variable objetivo en el tiempo cambia todo.** Los castigos recientes tienen LGD = 1 porque el cobro no ha terminado, y los créditos recientes con desenlace conocido son más malos porque solo terminaron los que cayeron pronto.
4. **Comparar grupos sin controlar el año engaña** cuando la población cambia con el tiempo: así pasó con la verificación del ingreso.
5. **Balancear las clases**, que parece la solución obvia con 19 % de malos, no mejoró el orden y dañó las probabilidades.

Sobre el proceso:

6. **La variabilidad entre semillas es parte del resultado.** Entre las mejores configuraciones de la búsqueda, la validación no alcanza a separar; una sola corrida habría hecho parecer real una diferencia que está dentro del ruido.
7. **No se decide con la prueba.** En la LGD, la configuración que eligió la validación le va peor en la prueba que la anterior. Cambiarla al verlo habría convertido la prueba en otra validación.
8. **Un método se pone a prueba en un caso donde se conoce la respuesta.** El reparto de puntos de la red reproduce exactamente los coeficientes cuando se aplica a la logística (sección 5.5), y eso da más confianza que revisar el código a ojo.
9. **Una aplicación que se entiende tiene que decir qué no se puede cambiar.** El "¿qué pasaría si...?" no ofrece cambiar variables que no dependen de la persona o cuya relación con el riesgo es de selección, porque la persona leería un consejo donde solo hay una correlación.
10. **Lo que se escribe a mano se desactualiza.** Numerar a mano casi 40 figuras y tablas escritas por tres personas no habría funcionado: el blog se numera solo al construirse y avisa si una figura no se cita (`python construir_blog.py`).

#### Referencias citadas en esta sección

Grinsztajn, L., Oyallon, E., y Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? En *Advances in Neural Information Processing Systems 35 (NeurIPS 2022), Datasets and Benchmarks Track*. https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html

ranadeep. (2021). *Credit risk analysis* [Conjunto de datos]. Kaggle. https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset

Shwartz-Ziv, R., y Armon, A. (2022). Tabular data: Deep learning is not all you need. *Information Fusion, 81*, 84–90. https://doi.org/10.1016/j.inffus.2021.11.011
