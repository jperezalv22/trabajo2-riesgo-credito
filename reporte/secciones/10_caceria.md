<!-- Sección 10. Autor: Sebastián (parte C). Sale de registro_ia.md: los hallazgos A1 a A8 se copian
     tal cual de allí; si cambia el registro, hay que volver a copiarlos.
     Numeración provisional: Tabla C1. -->

## 10. Cacería de la alucinación

Durante el trabajo anotamos cada error de la IA en el momento en que lo encontramos, en `registro_ia.md` del repositorio, con el prompt literal y la evidencia. Aquí están los ocho hallazgos confirmados (Tabla C1) y, al final, las verificaciones que no encontraron errores.

**Tabla C1.** Hallazgos de la cacería de la alucinación.

| # | Categoría | Qué dijo la IA | Cómo lo demostramos | Quién lo encontró |
|:--|:--|:--|:--|:--|
| A1 | Datos del mundo real / código | Que el 23 % de los castigados recupera "exactamente 0.14" de su exposición | Con igualdad exacta es el 0.2 %; el 23 % sale al redondear a dos decimales | Juan Pablo |
| A2 | Código | Que `rank(method="first")` reparte los empates para formar quintiles iguales | En H4 los quintiles 1 y 2 son todos 0 y tienen tasas distintas, porque el desempate sigue el orden del archivo, que va con la fecha | Santiago |
| A3 | Conceptos / datos | Que "verificado" indica más riesgo | Dentro de cada año y plazo la brecha baja de 7.6 a 2.9 puntos | Santiago |
| A4 | Código | Que `sys.modules["torch"] = None` sirve para probar el código sin PyTorch | La prueba se caía dentro de scipy; en un entorno real sin torch todo funciona | Santiago |
| A5 | Código | Que `feature_names_in_` son las columnas que usa el modelo | Lista también `id` y `malo`; la app recibía un aviso falso en cada llamada | Santiago |
| A6 | Datos del mundo real | Que red y logística "más se distancian" en `loan_amnt` y `revol_bal` | La diferencia en `plazo_meses` es casi el doble de la de `revol_bal` | Santiago |
| A7 | Código | `to_markdown` sobre textos ya formateados | tabulate los convertía en números: "+4.0" salía 4 | Santiago |
| A8 | Datos del mundo real | Que el modelo de la app se entrenó "con más de 200 000 créditos" | Son 130 832; el otro número incluye la prueba | Sebastián |

*Fuente: elaboración propia a partir de `registro_ia.md`. Todos los errores son de Claude Code, la herramienta que usamos los tres.*

Los ocho cubren tres de las cinco categorías del enunciado: código (A1, A2, A4, A5 y A7), datos del mundo real (A1, A3, A6 y A8) y conceptos (A3). No encontramos errores de matemáticas ni de bibliografía, aunque los buscamos: lo que revisamos está al final de la sección.

Si tuviéramos que proponer uno para el premio, sería **A2**. Es el más sutil: el código se ve correcto, el comentario explica bien lo que hace y el resultado parece razonable. El error solo aparece al preguntarse por qué dos grupos con el mismo valor de la variable tienen tasas de malos distintas, y la respuesta (el orden del archivo, que va con la fecha) es una forma de filtrar el tiempo en un análisis que no debía tenerlo.

### 10.1 Los hallazgos, uno por uno

#### A1. Una cifra redondeada que se volvió "exacta"

1. **Prompt literal** (2026-10-27, Claude Code): `de que trata el trabajo en terminos sencillos y facil de entender y de donde sale el dataseet lo escogiste tu o lo dio el profe`.
   Para responderlo, la IA exploró la LGD y corrió un cálculo que agrupaba lo recuperado sobre
   la EAD **redondeado a dos decimales**. La salida fue
   `recuperado/EAD mas comunes: {0.0: 0.39, 0.14: 0.23, 0.06: 0.103, 0.15: 0.06, ...}`.
2. **Lo que dijo la IA:** al resumir la conversación escribió *"the recovered fraction of EAD is
   0 in 39 % of cases, exactly 0.14 in 23 % and 0.06 in 10.3 %. This suggests a fixed recovery
   rule (possibly a debt sale)"*. El 2026-11-03 pasó la misma frase al `HANDOFF.md` ("exactamente
   0.14 en el 23 %") y se la repitió a Juan Pablo.
3. **Cómo se sospechó:** al escribir la sección 6 recalculamos el dato con `1 - lgd` y la
   igualdad exacta a 0.14 dio el 0.2 % de los castigados, no el 23 %.
4. **Evidencia:** buscamos en el historial el cálculo original y tenía `round(2)`. Con
   el redondeo, la recuperación bruta sobre la EAD cae en 0.14 en el 24.7 % de los castigados hasta
   2013 (`resultados/descriptivo/recuperaciones.md`). Con igualdad exacta, en el 0.2 %. La palabra
   "exactamente" la puso la IA al resumir; el código nunca la respaldó.
5. **Corrección:** la sección 6 dice "cerca del 14 %" y la Tabla 8 aclara que los valores están
   redondeados. El `HANDOFF.md` quedó corregido. La idea de una regla fija de recuperación sigue en pie como
   hipótesis (los picos se ven en la Figura 3), pero sin fuente primaria.
6. **Lección:** cuando una IA da una cifra con una precisión llamativa, hay que pedir el código
   que la produjo y revisar si redondea o agrupa. Cada resumen hecho por la IA (de una
   conversación, de un resultado) puede perder los matices como "aproximadamente".

#### A2. Quintiles que dependen del orden del archivo

1. **Prompt literal:** falta que Juan Pablo confirme cuál fue. El código aparece en el commit `8ebceb5` y el comentario en `998a304` (prompt 2 de la tabla).
2. **Lo que dijo la IA:** en `descriptivo.py`, `pd.qcut(serie.rank(method="first"), 5, ...)`, con el comentario *"rank() reparte los empates para que queden 5 grupos iguales"*.
3. **Cómo se sospechó:** al explorar los datos para la PD, los quintiles 1 y 2 de `inq_last_6mths` contienen solo el valor 0 y aun así tienen tasas de malos distintas.
4. **Evidencia:** `method="first"` desempata por la posición en el archivo, y esa posición está correlacionada con la fecha (0.73). El quintil 1 de H4 tiene año medio 2012.06 y 18.3 % de malos; el quintil 2, año medio 2013.09 y 20.1 %. La tasa real de `inq_last_6mths = 0` es 19.9 % (`df.groupby("inq_last_6mths")["malo"].mean()`). La Tabla 5 compara contra ese 18.3 % y la sección 3.2 dice que la tasa sube "de forma ordenada" de un quintil al siguiente, lo que en H4 es en parte un efecto del orden. H10 tiene el mismo problema con los empates en 10 años. Las conclusiones de H4 (sí) y H10 (no) no cambian: con 0 consultas hay 19.9 % de malos y con 2, 24.2 %.
5. **Corrección:** en la sección 5 comparamos valores concretos de la variable (percentiles 10 y 90), no quintiles. Queda pendiente que Juan Pablo decida si corrige `descriptivo.py` y la Tabla 5, por ejemplo agrupando por valor en las variables con muchos empates.
6. **Lección:** un desempate "para que queden grupos iguales" le agrega información ajena a la variable. Verificación que lo habría evitado: revisar el mínimo y el máximo de cada grupo.

#### A3. "Verificado indica más riesgo", sin controlar el tiempo

1. **Prompt literal:** el prompt 2 de la tabla (borrador de la sección 3).
2. **Lo que dijo la IA:** sección 3.3: *"dentro de cada quintil los verificados siguen siendo más riesgosos que los no verificados"*, y *"Para un modelo predictivo, "verificado" sí indica más riesgo y es correcto usarlo"*.
3. **Cómo se sospechó:** la dependencia parcial de los modelos da +0.1 puntos (red) y +0.4 (logística), y la variable casi no importa: su importancia por permutación es de 0.0007.
4. **Evidencia:** la Tabla 6 controla solo el monto. La verificación también cambia con los años, y los años recientes tienen más malos por el sesgo de madurez. Comparando dentro de cada año y plazo (`resultados/pd/verificacion_anio_plazo.md`), la brecha baja de 7.6 a 2.9 puntos; de 2010 a 2013 está entre −0.3 y +1.8 (salvo −5.0 a 60 meses en 2012), y solo en 2014-2015 es grande (de 4.3 a 9.1).
5. **Corrección:** la sección 5.6 lo explica. Queda pendiente que Juan Pablo ajuste la frase de la sección 3.3; la conclusión práctica sobre la aplicación (D12) no cambia.
6. **Lección:** cuando la población cambia con el tiempo, controlar una sola variable no basta para interpretar una diferencia. Verificación que lo habría evitado: repetir la comparación dentro de cada año.

#### A4. Una prueba que se caía por culpa de la prueba

1. **Prompt literal:** el prompt 3 (anexo), en la parte *"NO puede importar torch"*.
2. **Lo que dijo la IA:** en `pd_verificar.py`, `sys.modules["torch"] = None`, con el comentario *"Poner torch en None hace que cualquier "import torch" falle, incluso si lo intenta otra libreria"*.
3. **Cómo se sospechó:** la comprobación falló, pero el error estaba dentro de scipy y `riesgo/predecir_pd.py` no importa torch.
4. **Evidencia:** el error fue `AttributeError: 'NoneType' object has no attribute 'Tensor'` en `scipy/_lib/array_api_compat/common/_helpers.py`: scipy consulta esa entrada de `sys.modules` y se encuentra con el None. En un entorno creado sin torch (requirements.txt sin torch ni matplotlib), `evaluar_cliente` funcionó y `"torch" in sys.modules` dio False.
5. **Corrección:** un buscador en `sys.meta_path` que hace fallar solo `import torch`, más la comprobación de que torch no quedó cargado.
6. **Lección:** cuando una prueba falla, hay que leer dónde falla antes de creerle. Y la prueba que vale es la del entorno real: instalar sin torch.

#### A5. Un aviso falso en `evaluar_cliente`

1. **Prompt literal:** el prompt 3 (anexo), punto 5.
2. **Lo que dijo la IA:** en `riesgo/predecir_pd.py`, `columnas = list(preprocesador.feature_names_in_)`, tomándola como las columnas que el modelo necesita.
3. **Cómo se sospechó:** al probar un cliente sin `dti`, el aviso también pedía `id`, `loan_status` y `malo`.
4. **Evidencia:** el preprocesador se ajusta con el DataFrame completo, así que `feature_names_in_` tiene todas las columnas del parquet. Con ese código, la aplicación habría recibido un aviso en cada llamada.
5. **Corrección:** las columnas salen de los grupos de variables que usa el modelo, y `pd_verificar.py` convierte ese aviso en un error para un cliente completo.
6. **Lección:** probar también con entradas incompletas, no solo con una fila del conjunto de datos, que siempre trae todo.

#### A6. Una afirmación del borrador que la tabla no respaldaba

1. **Prompt literal:** el prompt 3 (anexo), punto 8.
2. **Lo que dijo la IA:** en el borrador de la sección 5.6: *"Donde más se distancian es en `loan_amnt` (caída de 0.0193 en la red y de 0.0112 en la logística) y `revol_bal` (0.0068 y 0.0036)"*.
3. **Cómo se sospechó:** al revisar una por una las cifras escritas a mano en el texto.
4. **Evidencia:** en `resultados/pd/importancia.csv`, las mayores diferencias son `loan_amnt` (+0.0081), `plazo_meses` (−0.0061) y `revol_bal` (+0.0033). La IA miró solo las diferencias a favor de la red.
5. **Corrección:** *"La red se apoya más que la logística en `loan_amnt` y en `revol_bal`, y menos en `plazo_meses`"*.
6. **Lección:** las frases con superlativos ("donde más") hay que calcularlas, no leerlas de una tabla a ojo.

#### A7. Tablas que perdían los signos

1. **Prompt literal:** el prompt 3 (anexo), punto 8 (*"Figuras y tablas"*).
2. **Lo que dijo la IA:** `tabla.to_markdown(index=False)`, sobre columnas con textos ya formateados como `"+4.0"`.
3. **Cómo se sospechó:** en `verificacion_anio_plazo.md`, la columna de brechas no tenía signos positivos y en `hipotesis_modelo.md` aparecía "-0".
4. **Evidencia:** tabulate intenta convertir los textos en números: `"+4.0"` salía como `4`, `"0.1800"` como `0.18` y `"-0.0"` como `-0`.
5. **Corrección:** `to_markdown(index=False, disable_numparse=True)` en las tablas de la PD. Las de la parte A siguen como estaban.
6. **Lección:** abrir el archivo generado, no solo lo que se imprime en la consola.

#### A8. Una cifra de la aplicación que nadie había calculado

1. **Prompt literal:** el prompt 4 (anexo), en la parte *"La persona llena sus datos [...] Que sea intuitiva para alguien que nunca la ha visto"*.
2. **Lo que dijo la IA:** en el texto de bienvenida de `app/app.py`: *"Esta aplicación le muestra lo que respondería un modelo entrenado con más de 200 000 créditos de Lending Club."*
3. **Cómo se sospechó:** al probar la app como usuario nuevo (con `streamlit.testing.v1.AppTest`) se releyeron sus textos contra las tablas del blog, y la sección 5 dice *"con 130 832 créditos para 90 parámetros"*.
4. **Evidencia:** `resultados/descriptivo/particiones_pd.md`: entrenamiento 130 832, validación 32 709 y prueba 104 989, en total 268 530 (Tabla 3). "Más de 200 000" no es ninguna partición: solo se llega allí sumando la prueba de 2014-2015, que el modelo nunca vio. El texto equivocado no alcanzó a llegar a un commit; quedó en la sesión de Claude Code del 2026-11-05.
5. **Corrección:** *"entrenado con 130 832 créditos de Lending Club, originados entre 2007 y 2013"* (commit `1f02206`).
6. **Lección:** toda cifra que lee el usuario debe salir de un archivo de resultados. Si la IA escribe un número redondo, hay que preguntar de dónde sale.

### 10.2 Verificaciones que no encontraron errores

El enunciado pide documentar también lo que se revisó sin encontrar nada. El detalle está en `registro_ia.md`; en resumen:

- **Matemáticas.** Revisamos paso a paso cada derivación del reporte: la scorecard y su inversa, el signo de los puntos de la logística, la pérdida fraccional de la LGD, los gradientes de la entropía cruzada y del error cuadrático con salida sigmoide, la retropropagación, la fórmula de Shapley y el número de parámetros de las redes. Las ecuaciones de Adam y RMSprop de la sección 4 se compararon con el código fuente de PyTorch 2.13.0. Las métricas se comprobaron a mano en un caso de cuatro créditos y contra `scipy` (`python pd_verificar.py`). No encontramos errores.
- **Bibliografía.** Antes de verificar las referencias de las secciones 1, 4 y 8, la IA las escribió de memoria en un archivo que guardamos en el repositorio (`reporte/verificacion/referencias_de_memoria.md`). Las 10 que tienen DOI coincidieron con Crossref en autores, año, volumen y páginas. Lo que no pudimos confirmar no se cita: las páginas de Nair y Hinton (2010) y un libro sobre scorecards cuyo año de publicación no coincide entre fuentes y que nadie del equipo leyó.
- **Afirmaciones sin fuente.** Quitamos antes de publicar varias frases que la IA escribió sin respaldo, aunque no demostramos que fueran falsas: el significado de los valores W y F de `initial_list_status`, que el diccionario no define; ejemplos de registros públicos que el diccionario no da; y una explicación de la hipótesis H6 escrita como un hecho, que ahora aparece como una explicación posible (sección 8.2).

Lo que aprendimos como equipo cabe en dos reglas. La primera: **ninguna cifra, referencia o definición pasa al reporte si no sale de un archivo del repositorio o de una fuente que alguien abrió.** Con ella se encontraron A1, A3, A6 y A8. La segunda: **ningún código se da por bueno sin mirar lo que produce con datos reales**: el archivo generado (A7), una entrada incompleta (A5), un entorno sin la librería (A4) o el mínimo y el máximo de cada grupo (A2).
