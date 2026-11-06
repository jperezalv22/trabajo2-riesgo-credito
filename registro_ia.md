# Registro de uso de IA

Aquí se anota **cada prompt importante en el momento en que se usa**, copiado literal, y cada
error de la IA que encontremos. De aquí salen la sección de uso de IA y la cacería de la
alucinación del blog. En el Trabajo 1 tocó reconstruir los prompts al final; aquí no.

## Prompts

| # | Fecha | Quién | Herramienta | Para qué | Prompt (literal) |
|---|---|---|---|---|---|
| 1 | 2026-10-26 | Juan Pablo | Claude Code | Plan del trabajo y base del repositorio (datos y LGD) | `Listo hora del taller dos, segun la experiencia con el uno, este texto y el enunciado en descargas cual es el plan para hacerlo? necesitas mas info? la idea es pensar en que hará cada uno y en que el codigo sea lo mas modular, facil de entender y humanizado posible, sin nada muy raro` |
| 2 | 2026-11-03 | Juan Pablo | Claude Code | Borradores de las secciones 2, 3 y 6 del blog y tablas reproducibles en `descriptivo.py` | Pregunta del asistente: `De tu parte A faltan las secciones del blog: datos, hipótesis y LGD (la de pérdida esperada necesita el modelo de PD de B). ¿Arranco ya con los borradores?` Respuesta: `Sí, arranca (Recomendado)` y luego `continua` |
| 3 | 2026-11-05 | Santiago | Claude Code | Parte B completa: métricas, logística, red, búsqueda, scorecard, predicción sin PyTorch, interpretabilidad y sección 5 del blog | Literal en el [anexo del prompt 3](#anexo-prompt-3-literal) |
| 4 | 2026-11-05 | Sebastián | Claude Code | Parte C completa: aplicación web, unión del blog, secciones 1, 4, 8, 9, 10 y 11, guion del video | Literal en el [anexo del prompt 4](#anexo-prompt-4-literal) |

## Posibles alucinaciones (anotar apenas aparezcan)

Para cada una: prompt literal, parte de la respuesta, cómo lo sospechamos, evidencia, corrección
y lección. Categorías: matemáticas, código, datos del mundo real, bibliografía, conceptos.

| # | Fecha | Quién | Categoría | Qué dijo la IA | Evidencia de que está mal | Estado |
|---|---|---|---|---|---|---|
| A1 | 2026-10-27 (detectada el 2026-11-03) | Juan Pablo | Datos del mundo real / código | Que el 23 % de los castigados recupera "exactamente 0.14" de su exposición | Con igualdad exacta es el 0.2 %; el 23 % sale al redondear a 2 decimales | Confirmada y corregida |
| A2 | 2026-11-05 | Santiago (revisa Juan Pablo) | Código | Los quintiles de `descriptivo.py`: "rank() reparte los empates para que queden 5 grupos iguales" | En H4 los quintiles 1 y 2 son todos `inq_last_6mths = 0`, y dan 18.3 % y 20.1 % de malos | Confirmada; sin corregir (es de la parte A) |
| A3 | 2026-11-05 | Santiago (revisa Juan Pablo) | Datos del mundo real / conceptos | Sección 3.3: los verificados son más riesgosos; "para un modelo predictivo, 'verificado' sí indica más riesgo" | Dentro de año y plazo la brecha baja de 7.6 a 2.9 puntos; el modelo le da +0.1 puntos | Confirmada; falta que Juan Pablo ajuste la sección 3.3 |
| A4 | 2026-11-05 | Santiago | Código | Que `sys.modules["torch"] = None` sirve para probar el código sin PyTorch | La prueba se cae dentro de scipy, no en nuestro código; en un entorno sin torch instalado todo funciona | Confirmada y corregida |
| A5 | 2026-11-05 | Santiago | Código | Usar `feature_names_in_` como las columnas que necesita el modelo | Lista 40 columnas, entre ellas `id` y `malo`; `evaluar_cliente` avisaba de datos que el modelo no usa | Confirmada y corregida |
| A6 | 2026-11-05 | Santiago | Datos del mundo real | Borrador de la sección 5: red y logística "más se distancian" en `loan_amnt` y `revol_bal` | La brecha en `plazo_meses` (0.0061) es casi el doble de la de `revol_bal` (0.0033) | Confirmada y corregida |
| A7 | 2026-11-05 | Santiago | Código | Escribir las tablas con `to_markdown` sobre textos ya formateados | tabulate los volvía a leer como números: "+4.0" salía 4, "0.1800" salía 0.18 y aparecía "-0" | Confirmada y corregida |
| A8 | 2026-11-05 | Sebastián | Datos del mundo real | Texto de la app: el modelo se entrenó "con más de 200 000 créditos" | El entrenamiento tiene 130 832 (Tabla 3); 268 530 es el total con la prueba | Confirmada y corregida |

### A1. Una cifra redondeada que se volvió "exacta"

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

### A2. Quintiles que dependen del orden del archivo

1. **Prompt literal:** falta que Juan Pablo confirme cuál fue. El código aparece en el commit `8ebceb5` y el comentario en `998a304` (prompt 2 de la tabla).
2. **Lo que dijo la IA:** en `descriptivo.py`, `pd.qcut(serie.rank(method="first"), 5, ...)`, con el comentario *"rank() reparte los empates para que queden 5 grupos iguales"*.
3. **Cómo se sospechó:** al explorar los datos para la PD, los quintiles 1 y 2 de `inq_last_6mths` contienen solo el valor 0 y aun así tienen tasas de malos distintas.
4. **Evidencia:** `method="first"` desempata por la posición en el archivo, y esa posición está correlacionada con la fecha (0.73). El quintil 1 de H4 tiene año medio 2012.06 y 18.3 % de malos; el quintil 2, año medio 2013.09 y 20.1 %. La tasa real de `inq_last_6mths = 0` es 19.9 % (`df.groupby("inq_last_6mths")["malo"].mean()`). La Tabla 5 compara contra ese 18.3 % y la sección 3.2 dice que la tasa sube "de forma ordenada" de un quintil al siguiente, lo que en H4 es en parte un efecto del orden. H10 tiene el mismo problema con los empates en 10 años. Las conclusiones de H4 (sí) y H10 (no) no cambian: con 0 consultas hay 19.9 % de malos y con 2, 24.2 %.
5. **Corrección:** en la sección 5 comparamos valores concretos de la variable (percentiles 10 y 90), no quintiles. Queda pendiente que Juan Pablo decida si corrige `descriptivo.py` y la Tabla 5, por ejemplo agrupando por valor en las variables con muchos empates.
6. **Lección:** un desempate "para que queden grupos iguales" le agrega información ajena a la variable. Verificación que lo habría evitado: revisar el mínimo y el máximo de cada grupo.

### A3. "Verificado indica más riesgo", sin controlar el tiempo

1. **Prompt literal:** el prompt 2 de la tabla (borrador de la sección 3).
2. **Lo que dijo la IA:** sección 3.3: *"dentro de cada quintil los verificados siguen siendo más riesgosos que los no verificados"*, y *"Para un modelo predictivo, "verificado" sí indica más riesgo y es correcto usarlo"*.
3. **Cómo se sospechó:** la dependencia parcial de los modelos da +0.1 puntos (red) y +0.4 (logística), y la variable casi no importa: su importancia por permutación es de 0.0007.
4. **Evidencia:** la Tabla 6 controla solo el monto. La verificación también cambia con los años, y los años recientes tienen más malos por el sesgo de madurez. Comparando dentro de cada año y plazo (`resultados/pd/verificacion_anio_plazo.md`), la brecha baja de 7.6 a 2.9 puntos; de 2010 a 2013 está entre −0.3 y +1.8 (salvo −5.0 a 60 meses en 2012), y solo en 2014-2015 es grande (de 4.3 a 9.1).
5. **Corrección:** la sección 5.6 lo explica. Queda pendiente que Juan Pablo ajuste la frase de la sección 3.3; la conclusión práctica sobre la aplicación (D12) no cambia.
6. **Lección:** cuando la población cambia con el tiempo, controlar una sola variable no basta para interpretar una diferencia. Verificación que lo habría evitado: repetir la comparación dentro de cada año.

### A4. Una prueba que se caía por culpa de la prueba

1. **Prompt literal:** el prompt 3 (anexo), en la parte *"NO puede importar torch"*.
2. **Lo que dijo la IA:** en `pd_verificar.py`, `sys.modules["torch"] = None`, con el comentario *"Poner torch en None hace que cualquier "import torch" falle, incluso si lo intenta otra libreria"*.
3. **Cómo se sospechó:** la comprobación falló, pero el error estaba dentro de scipy y `riesgo/predecir_pd.py` no importa torch.
4. **Evidencia:** el error fue `AttributeError: 'NoneType' object has no attribute 'Tensor'` en `scipy/_lib/array_api_compat/common/_helpers.py`: scipy consulta esa entrada de `sys.modules` y se encuentra con el None. En un entorno creado sin torch (requirements.txt sin torch ni matplotlib), `evaluar_cliente` funcionó y `"torch" in sys.modules` dio False.
5. **Corrección:** un buscador en `sys.meta_path` que hace fallar solo `import torch`, más la comprobación de que torch no quedó cargado.
6. **Lección:** cuando una prueba falla, hay que leer dónde falla antes de creerle. Y la prueba que vale es la del entorno real: instalar sin torch.

### A5. Un aviso falso en `evaluar_cliente`

1. **Prompt literal:** el prompt 3 (anexo), punto 5.
2. **Lo que dijo la IA:** en `riesgo/predecir_pd.py`, `columnas = list(preprocesador.feature_names_in_)`, tomándola como las columnas que el modelo necesita.
3. **Cómo se sospechó:** al probar un cliente sin `dti`, el aviso también pedía `id`, `loan_status` y `malo`.
4. **Evidencia:** el preprocesador se ajusta con el DataFrame completo, así que `feature_names_in_` tiene todas las columnas del parquet. Con ese código, la aplicación habría recibido un aviso en cada llamada.
5. **Corrección:** las columnas salen de los grupos de variables que usa el modelo, y `pd_verificar.py` convierte ese aviso en un error para un cliente completo.
6. **Lección:** probar también con entradas incompletas, no solo con una fila del conjunto de datos, que siempre trae todo.

### A6. Una afirmación del borrador que la tabla no respaldaba

1. **Prompt literal:** el prompt 3 (anexo), punto 8.
2. **Lo que dijo la IA:** en el borrador de la sección 5.6: *"Donde más se distancian es en `loan_amnt` (caída de 0.0193 en la red y de 0.0112 en la logística) y `revol_bal` (0.0068 y 0.0036)"*.
3. **Cómo se sospechó:** al revisar una por una las cifras escritas a mano en el texto.
4. **Evidencia:** en `resultados/pd/importancia.csv`, las mayores diferencias son `loan_amnt` (+0.0081), `plazo_meses` (−0.0061) y `revol_bal` (+0.0033). La IA miró solo las diferencias a favor de la red.
5. **Corrección:** *"La red se apoya más que la logística en `loan_amnt` y en `revol_bal`, y menos en `plazo_meses`"*.
6. **Lección:** las frases con superlativos ("donde más") hay que calcularlas, no leerlas de una tabla a ojo.

### A7. Tablas que perdían los signos

1. **Prompt literal:** el prompt 3 (anexo), punto 8 (*"Figuras y tablas"*).
2. **Lo que dijo la IA:** `tabla.to_markdown(index=False)`, sobre columnas con textos ya formateados como `"+4.0"`.
3. **Cómo se sospechó:** en `verificacion_anio_plazo.md`, la columna de brechas no tenía signos positivos y en `hipotesis_modelo.md` aparecía "-0".
4. **Evidencia:** tabulate intenta convertir los textos en números: `"+4.0"` salía como `4`, `"0.1800"` como `0.18` y `"-0.0"` como `-0`.
5. **Corrección:** `to_markdown(index=False, disable_numparse=True)` en las tablas de la PD. Las de la parte A siguen como estaban.
6. **Lección:** abrir el archivo generado, no solo lo que se imprime en la consola.

### A8. Una cifra de la aplicación que nadie había calculado

1. **Prompt literal:** el prompt 4 (anexo), en la parte *"La persona llena sus datos [...] Que sea intuitiva para alguien que nunca la ha visto"*.
2. **Lo que dijo la IA:** en el texto de bienvenida de `app/app.py`: *"Esta aplicación le muestra lo que respondería un modelo entrenado con más de 200 000 créditos de Lending Club."*
3. **Cómo se sospechó:** al probar la app como usuario nuevo (con `streamlit.testing.v1.AppTest`) se releyeron sus textos contra las tablas del blog, y la sección 5 dice *"con 130 832 créditos para 90 parámetros"*.
4. **Evidencia:** `resultados/descriptivo/particiones_pd.md`: entrenamiento 130 832, validación 32 709 y prueba 104 989, en total 268 530 (Tabla 3). "Más de 200 000" no es ninguna partición: solo se llega allí sumando la prueba de 2014-2015, que el modelo nunca vio. El texto equivocado no alcanzó a llegar a un commit; quedó en la sesión de Claude Code del 2026-11-05.
5. **Corrección:** *"entrenado con 130 832 créditos de Lending Club, originados entre 2007 y 2013"* (commit `1f02206`).
6. **Lección:** toda cifra que lee el usuario debe salir de un archivo de resultados. Si la IA escribe un número redondo, hay que preguntar de dónde sale.

## Verificaciones que no encontraron errores

- 2026-10-26 (Juan Pablo): las definiciones de las variables se tomaron del diccionario
  `LCDataDictionary.xlsx` del dataset, no de la IA. `verification_status` no aparece en la
  hoja LoanStats del diccionario (allí se llama `is_inc_v`); hay que citarla con ese nombre.
- 2026-11-03 (Juan Pablo): se revisaron contra los datos originales tres afirmaciones de la IA
  antes de ponerlas en el blog, y las tres se sostienen:
  - `Default` y `Late (31-120 days)` tienen `recoveries = 0` en el 100 % de los casos;
  - `tot_cur_bal`, `tot_coll_amt` y `total_rev_hi_lim` están vacías hasta 2011 (con un 52 % vacío
    en 2012);
  - los campos `*_joint` están vacíos en el 99.9 % y los `open_*`/`il_*` en el 97.6 %.
  Con esto se precisó la decisión D3, que decía "solo existen desde 2012".
- 2026-11-03 (Juan Pablo): referencias verificadas en la fuente.
  - Papke y Wooldridge (1996): Crossref y el DOI resuelve a Wiley.
  - Bergstra y Bengio (2012): página de JMLR, 13(10):281−305.
  - Conjunto de Kaggle: metadatos de la API, última actualización el 2021-06-12, usuario ranadeep.
- 2026-11-05 (Santiago): lo de la PD se comprobó contra resultados conocidos. Se puede repetir con
  `python pd_verificar.py`.
  - AUC, Gini, KS y Brier a mano en un caso de 4 créditos. AUC contra el conteo de Mann-Whitney y
    KS contra `scipy.stats.ks_2samp` con los datos reales.
  - Scorecard: 600 puntos con odds de 4, +20 al duplicarlas, −20 al reducirlas a la mitad, y
    `probabilidad(puntaje(p)) = p`.
  - Los puntos suman el puntaje (error máximo de 1e-5). Aplicado a la logística, el algoritmo de
    permutaciones da lo mismo que los coeficientes (error máximo de 2e-7).
  - `predecir_numpy` contra PyTorch: diferencia máxima de 1.2e-7.
  - Reproducibilidad: el intento 3 de la búsqueda y la semilla 2026 del entrenamiento son la misma
    configuración y dan exactamente la misma pérdida en validación (0.4500561952590942).
  - `python pd_entrenar.py` en un clon limpio, en la misma máquina: las métricas de las 30 semillas,
    los modelos y las tablas salen idénticos. Solo cambia el tiempo (525 s y 601 s).
  - Agregar el optimizador a `riesgo/red.py` no cambió la LGD: la demo da los mismos números.
- 2026-11-05 (Santiago): `weight_decay` en Adam se comprobó en el código de PyTorch instalado
  (`torch/optim/adam.py`). Por defecto suma `weight_decay * param` al gradiente de todos los
  parámetros (`decoupled_weight_decay=False`): es L2 con penalización (λ/2)‖θ‖², no AdamW.
- 2026-11-05 (Santiago): referencias de la sección 5, verificadas en la fuente.
  - Crossref, con el DOI resolviendo en doi.org: Breiman (2001), Brier (1950), Fawcett (2006),
    Friedman (2001), Shapley (1953), Štrumbelj y Kononenko (2014) y van den Goorbergh et al. (2022).
  - Fisher et al. (2019): página de JMLR, 20(177):1−81. Kingma y Ba (2015): página de arXiv.
  - van den Goorbergh et al.: se leyó el resumen en arXiv (2202.09101) y dice lo que se les atribuye.
  - RMSprop: las diapositivas dicen "Geoffrey Hinton with Nitish Srivastava, Kevin Swersky", Lecture
    6e. Por eso se citan así y no como "Tieleman y Hinton (2012)", la forma en que se suele citar.
  - SAS Enterprise Miner: se leyó la página y dice lo que se le atribuye (600 puntos con 30:1, 20
    para duplicar).
  - Štrumbelj y Kononenko: Springer no dio acceso al texto. Que muestrean el orden y una instancia de
    referencia se confirmó solo en una fuente secundaria (Molnar, *Interpretable Machine Learning*).
    Quien tenga acceso debería confirmarlo en el artículo.
  - Shapley (1953): Crossref da las páginas 307-318; de memoria la IA tenía 307-317. Se dejó la de
    Crossref.
  - Siddiqi, *Credit Risk Scorecards*: **no se cita.** La IA lo recordaba de 2006, Crossref dice 2012
    y OpenLibrary, 2005, para el mismo ISBN. Además nadie del equipo ha leído el libro.
- 2026-11-05 (Santiago): los colores de las figuras se revisaron con un validador de daltonismo.
  Terracota y verde, los de la Figura 4 de la LGD, casi no se distinguen con daltonismo protán (ΔE
  de 4.6, cuando el mínimo aceptable es 8). En las figuras de la PD se usó terracota y azul (ΔE de 20).

- 2026-11-05 (Sebastián, parte C): referencias de las secciones 1, 4 y 8.
  - La IA escribió primero las referencias de memoria en `reporte/verificacion/referencias_de_memoria.md`
    (commit `7df7c49`) y después se compararon con la fuente. Las 10 con DOI coinciden en Crossref en
    autores, año, volumen, número y páginas: McCulloch y Pitts (1943), Rosenblatt (1958), Rumelhart et
    al. (1986), Hornik et al. (1989), Prechelt (1998, editores Orr y Müller según el DOI del libro),
    Hanley y McNeil (1982), Massey (1951), Hand y Henley (1997), Lessmann et al. (2015) y Shwartz-Ziv y
    Armon (2022). Hand y Henley y Lessmann et al. al final no se citan, así que no van en el blog.
  - Srivastava et al. (2014): página de JMLR, 15(56):1929-1958. Loshchilov y Hutter: arXiv 1711.05101
    dice "Published as a conference paper at ICLR 2019"; la sede (Nueva Orleans) se confirmó en iclr.cc.
  - Grinsztajn et al. (2022): el título cambia entre arXiv (*"...on tabular data?"*) y NeurIPS
    (*"...on typical tabular data?"*). Se cita la versión de NeurIPS, cuya página se abrió.
  - Nair y Hinton (2010): se abrió el PDF en icml.cc. Las páginas 807-814 que recordaba la IA no se
    pudieron confirmar (ACM no respondió), así que se cita sin páginas.
  - Massey (1951): al escribir la sección 4 la IA agregó "Jr." al autor, que no está en Crossref ni en
    su versión de memoria. Se quitó. No se demostró que esté mal, pero no tiene respaldo.
  - BCBS (2006): se leyeron los párrafos 375 y 376 del PDF. Llaman "EL" a PD x LGD y "EL amount" a EL x EAD.
  - Lo que dicen Grinsztajn et al. y Shwartz-Ziv y Armon se tomó de sus resúmenes en arXiv. Grinsztajn
    et al. hablan de datos de tamaño medio (~10 000 registros), por eso la sección 8 no extiende su
    conclusión a nuestros 130 832.
- 2026-11-05 (Sebastián): fórmulas de la sección 4, contra el código fuente de PyTorch v2.13.0
  (`torch/optim/adam.py` y `rmsprop.py` en GitHub, porque torch no estaba instalado en esta máquina).
  - Adam: `denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)` y paso `lr / bias_correction1`,
    que es alpha * m_hat / (sqrt(v_hat) + eps). Betas (0.9, 0.999) y eps 1e-8 por defecto. Coincide.
  - RMSprop: `square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)` con alpha = 0.99 por defecto.
    Las diapositivas de Hinton dicen "MeanSquare(w, t) = 0.9 MeanSquare(w, t-1) + 0.1 (dE/dw)^2". Los
    dos valores salen así en la sección 4.6.
  - Las derivaciones de las secciones 5 y 6 se revisaron paso a paso y están bien: scorecard
    (B = 28.85 y A = 560), su inversa, el signo de los puntos de la logística, la referencia
    pi(1 - pi) del Brier, la derivada de la pérdida fraccional (se anula en p = m), el gradiente del
    error cuadrático con sigmoide y la fórmula de Shapley. Los parámetros de las redes (19 841 y 7 105)
    coinciden con la fórmula de la sección 4.2.
- 2026-11-05 (Sebastián): textos de la app y del blog que la IA escribió sin fuente y se quitaron
  antes del commit. No se cuentan como alucinaciones porque no se demostró que estuvieran mal:
  - `initial_list_status`: la IA escribió "completo (w) o fraccionado (f)". El diccionario solo dice
    *"Possible values are – W, F"*; la otra hoja menciona una "wholeloan platform", que no basta.
  - `pub_rec`: la IA agregó "por ejemplo, quiebras o embargos". El diccionario dice *"Number of
    derogatory public records"*.
  - Sección 4: "el Gini es el que más se usa en riesgo de crédito" (sin fuente) y una frase que le
    atribuía a Prechelt (1998) por qué la parada temprana regulariza, cuando su capítulo trata de los
    criterios para parar.
  - Sección 8: la explicación de H6 ("quien tiene una historia más larga suele tener más ingreso y
    más cuentas") estaba escrita como un hecho. Se calcularon las correlaciones de Spearman hasta 2013
    (0.41 con `total_acc` y 0.31 con `annual_inc`) y quedó escrita como una explicación posible.
- 2026-11-05 (Sebastián): errores de operación del asistente, sin efecto en los resultados. Un
  `pkill -f "pip install ..."` cerró su propia terminal, porque el patrón también aparecía en su
  propia línea de comandos (salida 144), y la sección 4 citaba la "sección 6.3" para algo que está en la 6.2.

## Anexo: prompt 3 literal

Se copió tal cual llegó al asistente. El texto llegó con el primer bloque repetido y con pedazos
perdidos al pegar (por ejemplo `https://github.com/jpedito` o `pd_puns del conjunto`). Lo que el
asistente tuvo que suponer por esos cortes está en el pull request de la parte B.

```text
Eres mi asistente para el Trabajo 2 de Redes Neuronales y Algoritmos Bioinspirados (UNAL 2026-2).
Te adjunto el enunciado en PDF: léelo completo antes de empezar.

Somos 3: Juan Pablo (parte A: datos y LGD, ya hecha), yo (parte B: probabilidad de incumplimiento,
PD) y otro compañero (parte C: app web, blog y video). Juan Pablo revisa y fusiona todo al final.

Plazo: entregamos esta semana (antes del domingo 8 de noviembre). Abre el pull request a más
tardar el jueves 5 de noviembre en la noche, aunque falten cosas, para alcanzar a integrar.

Eres mi asistente para el Trabajo 2 de Redes Neuronales y Algoritmos Bioinspirados (UNAL 2026-2).
Te adjunto el enunciado en PDF: léelo completo antes de empezar.

Somos 3: Juan Pablo (parte A: datos y LGD, ya hecha), yo (parte B: probabilidad de incumplimiento,
PD) y otro compañero (parte C: app web, blog y video). Juan Pablo revisa y fusiona todo al final.

Plazo: entregamos esta semana (antes del domingo 8 de noviembre). Abre el pull request a más
tardar el jueves 5 de noviembre en la noche, aunque falten cosas, para alcanzar a integrar.

## Preparación
1. git clone https://github.com/jpedito y crea la rama parte-b-pd.
2. Crea un entorno e instala requirements.txt. Si alguna versión no instala en mi Python, dímelo
   antes de cambiarla.
3. Lee HANDOFF.md (decisiones D1 a D11: NO las cambies sin anotarlo ahí), README.md, config.yaml,
   todo riesgo/ y lgd_entrenar.py, que es el ejemplo de estilo a imitar.
4. Corre python lgd_entrenar.py --demo para comprobar que todo funciona.

## Qué ya existe (reutilízalo, no lo reescribas)
- config.yaml + riesgo/configuraciel config y se cambia con
  --set seccion.parametro=valor. Cada script tiene --demo (muestra del 10 %, < 2 minutos).
- riesgo/datos.py: datos.cargar(carpeta, "pd", fraccion, semilla) -> 268 530 créditos, columna
malo.
- riesgo/particiones.py: entrenamiento/validación <= 2013 (80/20 estratificado), prueba temporal 2014-2015.
- riesgo/preparar.py: el MISMO preica y la red (comparación justa).
- riesgo/red.py: crear_red, entrenar (Adam + parada temprana + L2 + dropout), predecir, guardar,
  cargar, contar_parametros y exportar_numpy.
- riesgo/red_numpy.py: predecir_numpy, la misma predicción de la red pero solo con NumPy (sin torch).
- No modifiques riesgo/datos.py, particiones.py ni preparar.py: cambian los resultados de la LGD.
  Si necesitas algo nuevo en red.pn romper lo que ya hay.

## Mi parte (B)
1. Métricas en riesgo/metricas.py: (máximo de |TPR − FPR|), Brier
   y curva de calibración por deciles. Funciones cortas, cada una con su definición en el docstring.
2. pd_entrenar.py:
   - Regresión logística (modelo de referencia), con un ajuste pequeño de C en validación.
     Reporta cuántas configuraciones probaste.
   - Red neuronal (perceptrón multicapa) con la configuración final, entrenada con 30 semillas
     (semilla + k). Reporta media, desviación, mejor y peor de AUC, Gini, KS y Brier en prueba,
     el número de parámetros entrenables de cada modelo y el tiempo total.
   - Sin pesos por clase ni sobremuestreo, salvo que se justifique y se recalibre: dañan la
calibración.
   - Guarda en modelos/: pd_preprocesador.joblib, pd_logistica.joblib, pd_red.pt, pd_red.npz
     (con exportar_numpy) y pd_puns del conjunto de entrenamiento,
     para el percentil de la app).
3. pd_busqueda.py: búsqueda aleatoria (Bergstra y Bengio, 2012) de capas, neuronas, activación,
   dropout, L2, tasa de aprendizaje y tamaño de lote. La rejilla y el número de intentos van en
   config.yaml (sección pd.busqueda). Se elige por validación, nunca por prueba. Reporta cuántas
   configuraciones y cuánto tiempo
4. Scorecard en riesgo/scorecard.py. Parámetros en config.yaml, sección scorecard:
   puntaje_base, odds_referencia y pdo.
   - odds = (1 − p) / p (buenos:ma
     offset = puntaje_base − factor · ln(odds_referencia), puntaje = offset + factor · ln(odds).
   - Funciones puntaje(p) y probabilidad(puntaje).
   - Decide cómo sacar los puntos l modelo (con la logística salen
     directos de los coeficientes; con la red hay que justificar el método) y explícamelo.
5. riesgo/predecir_pd.py, que es lo que usa la app. NO puede importar torch (usa predecir_numpy
   de riesgo/red_numpy.py). Esta i porque el de la parte C ya la estáusando:                                                                                               cargar(carpeta="modelos") ->
     probabilidad(modelos, df) -> np.ndarray   (df con las columnas de datos/procesados/pd.parquet)
     evaluar_cliente(modelos, cliente: dict) -> {"pd": float, "puntaje": float,
                                                  "percentil": float, "puntos": {variable:       puntos}}
6. pd_interpretar.py: qué variables hacen más riesgosa a una persona, con un método justificado y
   citado (importancia por permutación o SHAP). Compáralo con los coeficientes de la logística y con
   las hipótesis H1 a H10 de resultados/descriptivo/hipotesis.md: cuáles confirma el modelo.
7. Figuras en resultados/pd/: curva ROC, calibración, distribución del puntaje de buenos y malos,
   e importancia de variables. Eje
8. reporte/secciones/05_pd.md: el texto del blog de mi parte (modelos, búsqueda, tabla con las 30
   semillas, derivación matemática de la scorecard, interpretabilidad y qué aportó la red frente a
   la logística, con métricas, varabilidad). Figuras y tablas contítulo,
   fuente y citadas en el texto ("como se ve en la Figura X"). El compañero C las renumera al final.
   Recuerda el sesgo de madurez de la prueba 2014-2015 (HANDOFF D5).

## Reglas
- Código simple, fácil de explicares cortas, nombres en español, sin
  clases complicadas ni librerías raras (nada de AutoML ni Optuna). Comentarios solo para el "por qué".
- Un script = un experimento, que . Semilla desde config.yaml.
- Lo que se imprime en consola sin símbolos raros (≥, ≤, −): la consola de Windows se cae con ellos.
- Commits pequeños y frecuentes desde MI cuenta de GitHub. Al terminar, abre un pull request de
  parte-b-pd a main (no fusion
- registro_ia.md: anota cada prompt importante LITERAL, con fecha y para qué. Cada vez que la IA
  (tú incluido) se equivoque en una derivada, una métrica, un dato o una referencia, anótalo con:
  el prompt, lo que dijo, cómo se sospechó, la evidencia (derivación, experimento o fuente
  primaria), la corrección y la lección. Vale el 15 % de la nota.
- Si no estás seguro de algo, dilo y verifícalo. Nunca inventes una referencia, un DOI, el
  significado de una variable (se citan de datos/diccionario/LCDataDictionary.xlsx) ni un resultado.
- Explícame cada decisión en palabras simples: el profesor me la va a preguntar a mí, en vivo.

## Listo cuando
- python pd_entrenar.py --demo c python pd_entrenar.py completofunciona.
- Los modelos están en modelos/, evaluar_cliente funciona sin torch y la sección 05_pd.md está escrita.
- El pull request está abierto, con un resumen de los resultados y de lo que falta.
```

## Anexo: prompt 4 literal

Se copió tal cual llegó al asistente. Como el 3, llegó con un pedazo repetido al pegar (desde
"README.md: un comando para cada experimento" hasta "Explícame cada decisión") y con la primera
copia de "Explícame cada decisión en palabras simples:" cortada. El PDF del enunciado no venía
adjunto: el asistente lo buscó en la carpeta de trabajo después de que Sebastián escribió
`ya descarge el pdf del trabajo`.

```text
Eres mi asistente para el Trabajo 2 de Redes Neuronales y Algoritmos Bioinspirados (UNAL 2026-2).
Te adjunto el enunciado en PDF: léelo completo antes de empezar.

Somos 3: Juan Pablo (parte A: datos y LGD, ya hecha), otro compañero (parte B: modelo de PD y
scorecard) y yo (parte C: aplicación web, blog y video promocional). Juan Pablo revisa y fusiona
todo al final.

Plazo: entregamos esta semana (antes del domingo 8 de noviembre). Abre el pull request a más
tardar el jueves 5 de noviembre en la noche, aunque falten cosas, para alcanzar a integrar.

## Preparación
1. git clone https://github.com/jperezalv22/trabajo2-riesgo-credito y crea la rama parte-c-app-blog.
2. Crea un entorno e instala requirements.txt. Si alguna versión no instala, dímelo antes de cambiarla.
3. Lee HANDOFF.md (decisiones D1 a D11), README.md, config.yaml, riesgo/ y lgd_entrenar.py.
   Mira resultados/descriptivo/ y resultados/lgd/, que ya tienen tablas y figuras.

## Mi parte (C)

### 1. Aplicación web (app/app.py, con Streamlit)
- Se levanta con un solo comando: streamlit run app/app.py.
- app/requirements.txt solo con lo que necesita la app (streamlit, numpy, pandas, scikit-learn,
  joblib, pyarrow, pyyaml) y SIN torch: la red se evalúa con NumPy (riesgo/red_numpy.py, predecir_numpy).
- La persona llena sus datos (las variables de variables en config.yaml) con etiquetas en español
  sencillo y valores por defecto razonables (mediana o categoría más común de datos/procesados/pd.parquet).
- Muestra:
  - su puntaje y su probabilidad de incumplimiento;
  - su percentil frente a la población (histograma de modelos/pd_puntajes_poblacion.npy con su posición);
  - "¿qué pasaría si...?": que elija una característica, la cambie y vea el puntaje nuevo y la diferencia;
  - los enlaces al blog y al video promocional.
- Usa SOLO esta interfaz de riesgo/predecir_pd.py, que escribe el compañero B:
    cargar(carpeta="modelos") -> dict
    evaluar_cliente(modelos, cliente: dict) -> {"pd": float, "puntaje": float,
                                                 "percentil": float, "puntos": {variable: puntos}}
  Mientras B no la suba, crea app/stub_predecir.py con la misma interfaz y valores de mentira, y
  bórralo antes del pull request.
- Que sea intuitiva para alguien que nunca la ha visto. Pruébala tú como usuario nuevo.
- Publicación: Streamlit Community Cloud (u otro servicio gratuito) desde el repositorio. Si pide
  permisos de dueño del repo, avísame para que lo haga Juan Pablo.

### 2. Blog (reporte técnico publicado)
- Cada dueño escribe su parte en reporte/secciones/:
  01_introduccion_caso_de_uso.md (yo, con Juan Pablo), 02_datos.md y 03_descriptivo_hipotesis.md
  (Juan Pablo), 04_fundamentos.md (yo), 05_pd.md (B), 06_lgd.md y 07_perdida_esperada.md
  (Juan Pablo), 08_discusion_aprendizajes.md (yo, con todos), 09_uso_ia.md, 10_caceria.md y
  11_referencias.md (yo).
- construir_blog.py une las secciones en reporte/blog.md y genera docs/index.html (con pandoc y
  MathJax para las fórmulas) para GitHub Pages. Juan Pablo activa Pages porque es el dueño.
  Referencia del estilo del trabajo anterior: https://jperezalv22.github.io/redes-neuronales-1/
- 04_fundamentos.md, con todo lo que pide el enunciado: neurona artificial, perceptrón multicapa,
  cada activación usada (ReLU, tanh, sigmoide), derivación de cada pérdida por máxima verosimilitud
  (entropía cruzada binaria desde Bernoulli, error cuadrático desde Gauss), retropropagación,
  ecuaciones de Adam, formulación de L2, dropout y parada temprana, y la definición de cada métrica
  (AUC, Gini, KS, Brier, MAE, RMSE). Revisa cada derivación paso a paso: la IA se equivoca con los
  signos y los factores. Si encuentras un error, va a la cacería.
- Todas las figuras y tablas numeradas en orden, con título y fuente, y citadas en el texto
  ("como se observa en la Figura 3"). Ninguna figura huérfana. Tú haces la numeración final.
- 09_uso_ia.md y 10_caceria.md salen de registro_ia.md. Al menos 5 hallazgos reales que cubran al
  menos 3 categorías (matemáticas, código, datos, bibliografía, conceptos), cada uno con: prompt
  literal, respuesta, cómo se sospechó, evidencia, corrección y lección. NO inventes hallazgos:
  uno inventado se penaliza más que uno faltante.
- 11_referencias.md en APA 7: solo referencias que existan, verificadas (abre cada DOI) y citadas
  en el texto.
- Enlaces al repositorio, a la app y al video.

### 3. Video promocional (2 a 3 minutos)
- Guion que presente la app como la solución a un problema y genere ganas de usarla. Tienen que
  aparecer los tres. Súbelo a YouTube como no listado y pon el enlace en la app y en el blog.

### 4. Al final
- README.md: un comando para cada experimento, la app y el blog.
- Probar todo en un clon limpio, en una máquina distinta a la de Juan Pablo.

## Reglas
- Código simple, fácil de explicar en la sustentación: funciones cortas, nombres en español, sin
  librerías raras. Comentarios solo para el "por qué". Parámetros en config.yaml, no en el código.
- Lo que se imprime en consola sin símbolos raros (≥, ≤, −): la consola de Windows se cae con ellos.
- Commits pequeños y frecuentes desde MI cuenta de GitHub. Al terminar, abre un pull request de
  parte-c-app-blog a main (no fusiones tú).
- registro_ia.md: anota cada prompt importante LITERAL, con fecha y para qué, y cada error de la IA
  (tú incluido), con evidencia.
- Si no estás seguro de algo, dilo y verifícalo con una fuente primaria. Nunca inventes una
  referencia, un DOI, una fórmula ni el significado de una variable.
- Explícame cada decisión en palabras simples:
- La app corre con streamlit run app/app.py y está publicada, el blog se construye con un comando
  y las secciones 01, 04, 08, 09, 10 y 11 están escritas.
- El guion del video está listo y el pull request está abierto, con un resumen de lo que falta.

---
- README.md: un comando para cada experimento, la app y el blog.
- Probar todo en un clon limpio, en una máquina distinta a la de Juan Pablo.

## Reglas
- Código simple, fácil de explicar en la sustentación: funciones cortas, nombres en español, sin
  librerías raras. Comentarios solo para el "por qué". Parámetros en config.yaml, no en el código.
- Lo que se imprime en consola sin símbolos raros (≥, ≤, −): la consola de Windows se cae con ellos.
- Commits pequeños y frecuentes desde MI cuenta de GitHub. Al terminar, abre un pull request de
  parte-c-app-blog a main (no fusiones tú).
- registro_ia.md: anota cada prompt importante LITERAL, con fecha y para qué, y cada error de la IA
  (tú incluido), con evidencia.
- Si no estás seguro de algo, dilo y verifícalo con una fuente primaria. Nunca inventes una
  referencia, un DOI, una fórmula ni el significado de una variable.
- Explícame cada decisión en palabras simples: el profesor me la puede preguntar a mí, en vivo.

## Listo cuando
- La app corre con streamlit run app/app.py y está publicada, el blog se construye con un comando
  y las secciones 01, 04, 08, 09, 10 y 11 están escritas.
- El guion del video está listo y el pull request está abierto, con un resumen de lo que falta.
```
