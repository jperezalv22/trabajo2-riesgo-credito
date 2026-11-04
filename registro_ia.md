# Registro de uso de IA

Aquí se anota **cada prompt importante en el momento en que se usa**, copiado literal, y cada
error de la IA que encontremos. De aquí salen la sección de uso de IA y la cacería de la
alucinación del blog. En el Trabajo 1 tocó reconstruir los prompts al final; aquí no.

## Prompts

| # | Fecha | Quién | Herramienta | Para qué | Prompt (literal) |
|---|---|---|---|---|---|
| 1 | 2026-10-26 | Juan Pablo | Claude Code | Plan del trabajo y base del repositorio (datos y LGD) | `Listo hora del taller dos, segun la experiencia con el uno, este texto y el enunciado en descargas cual es el plan para hacerlo? necesitas mas info? la idea es pensar en que hará cada uno y en que el codigo sea lo mas modular, facil de entender y humanizado posible, sin nada muy raro` |
| 2 | 2026-11-03 | Juan Pablo | Claude Code | Borradores de las secciones 2, 3 y 6 del blog y tablas reproducibles en `descriptivo.py` | Pregunta del asistente: `De tu parte A faltan las secciones del blog: datos, hipótesis y LGD (la de pérdida esperada necesita el modelo de PD de B). ¿Arranco ya con los borradores?` Respuesta: `Sí, arranca (Recomendado)` y luego `continua` |

## Posibles alucinaciones (anotar apenas aparezcan)

Para cada una: prompt literal, parte de la respuesta, cómo lo sospechamos, evidencia, corrección
y lección. Categorías: matemáticas, código, datos del mundo real, bibliografía, conceptos.

| # | Fecha | Quién | Categoría | Qué dijo la IA | Evidencia de que está mal | Estado |
|---|---|---|---|---|---|---|
| A1 | 2026-10-27 (detectada el 2026-11-03) | Juan Pablo | Datos del mundo real / código | Que el 23 % de los castigados recupera "exactamente 0.14" de su exposición | Con igualdad exacta es el 0.2 %; el 23 % sale al redondear a 2 decimales | Confirmada y corregida |

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
