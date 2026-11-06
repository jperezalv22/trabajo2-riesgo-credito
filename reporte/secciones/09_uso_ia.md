<!-- Sección 9. Autor: Sebastián (parte C). Sale de registro_ia.md (tabla de prompts y anexos).
     Numeración provisional: Tabla U1. -->

## 9. Uso de inteligencia artificial

Los tres usamos Claude Code, un asistente de programación con inteligencia artificial que trabaja dentro del repositorio: lee los archivos, corre los scripts y propone cambios que se revisan antes de cada commit. Desde el primer día anotamos en `registro_ia.md` cada prompt importante en el momento en que se usó, copiado literal, y cada error de la IA que encontramos. En el Trabajo 1 tuvimos que reconstruir los prompts al final, y esta vez no quisimos repetirlo.

### 9.1 Los prompts principales

La Tabla U1 resume los cuatro prompts que definieron el trabajo. Los dos primeros son cortos y van completos. Los prompts 3 y 4 son las instrucciones de las partes B y C: ocupan cerca de 80 líneas cada uno y están completos, tal como se pegaron, en los anexos de `registro_ia.md`. Aquí citamos las partes que más pesaron en el resultado.

**Tabla U1.** Prompts principales del trabajo.

| # | Fecha | Quién | Para qué | Prompt (literal o extracto literal) |
|:--|:--|:--|:--|:--|
| 1 | 2026-10-26 | Juan Pablo | Plan del trabajo y base del repositorio (datos y LGD) | `Listo hora del taller dos, segun la experiencia con el uno, este texto y el enunciado en descargas cual es el plan para hacerlo? necesitas mas info? la idea es pensar en que hará cada uno y en que el codigo sea lo mas modular, facil de entender y humanizado posible, sin nada muy raro` |
| 2 | 2026-11-03 | Juan Pablo | Borradores de las secciones 2, 3 y 6 | Pregunta del asistente: `De tu parte A faltan las secciones del blog: datos, hipótesis y LGD (la de pérdida esperada necesita el modelo de PD de B). ¿Arranco ya con los borradores?` Respuesta: `Sí, arranca (Recomendado)` y luego `continua` |
| 3 | 2026-11-05 | Santiago | Parte B: PD, scorecard, interpretabilidad y sección 5 | Extracto: `Cada vez que la IA (tú incluido) se equivoque en una derivada, una métrica, un dato o una referencia, anótalo con: el prompt, lo que dijo, cómo se sospechó, la evidencia (derivación, experimento o fuente primaria), la corrección y la lección.` |
| 4 | 2026-11-05 | Sebastián | Parte C: aplicación, unión del blog, secciones 1, 4, 8 a 11 y guion del video | Extracto: `Revisa cada derivación paso a paso: la IA se equivoca con los signos y los factores. Si encuentras un error, va a la cacería.` y `NO inventes hallazgos: uno inventado se penaliza más que uno faltante.` |

*Fuente: `registro_ia.md` del repositorio, con los prompts 3 y 4 completos en sus anexos.*

Los cuatro tienen algo en común: no piden solo un resultado, sino también cómo comprobarlo. El primero pidió desde el comienzo código "fácil de entender y humanizado", y de ahí salieron las funciones cortas con nombres en español y la regla de poner los parámetros en `config.yaml`. El 3 y el 4 convirtieron la revisión en una tarea explícita del asistente: anotar sus propios errores, comprobar las fórmulas contra resultados conocidos y no citar nada que no se hubiera abierto.

### 9.2 Qué hizo la IA y qué hicimos nosotros

- **Código.** La IA escribió la primera versión de casi todo el código: la preparación de los datos, el bucle de entrenamiento, la scorecard, la aplicación y el script que arma este blog. Nosotros decidimos la estructura, revisamos cada cambio y lo comprobamos con datos. Varios errores solo aparecieron así: con una entrada incompleta (A5), al abrir el archivo generado (A7) o al mirar dos grupos que debían ser iguales (A2).
- **Decisiones de modelado.** Las decisiones que se discuten en el reporte (qué variables excluir, cómo partir los datos, cómo definir la LGD, no balancear las clases, no recalibrar con la prueba) se discutieron con la IA, pero cada una quedó escrita con su razón en `HANDOFF.md` y la tomamos nosotros. Más de una vez la IA propuso algo razonable que los datos no respaldaron. Por ejemplo, la sección 3.3 decía que "verificado" indica más riesgo, y al comparar dentro de cada año y plazo la brecha baja de 7.6 a 2.9 puntos (A3).
- **Texto.** La IA escribió los borradores de las secciones a partir de las tablas de resultados, y cada autor los revisó contra esas tablas. Ahí se encontraron dos de los ocho errores (A6 y A8): frases y cifras que sonaban bien y que ninguna tabla respaldaba.
- **Matemáticas y referencias.** Las derivaciones de la sección 4 las escribió la IA y las revisamos paso a paso. Las ecuaciones de los optimizadores se compararon con el código de PyTorch. Las referencias las escribió primero de memoria, en un archivo que guardamos, y después las comparamos con Crossref, doi.org y las páginas de cada revista.

### 9.3 Impacto en el resultado final

El efecto más visible fue de tiempo: con el asistente, en diez días, tres personas escribieron un repositorio con diez scripts que corren con un solo comando cada uno, una aplicación web y este reporte. Pero el efecto más importante fue de otro tipo. La IA escribe con la misma seguridad cuando tiene razón y cuando no, así que hubo que tratar todo lo que producía como un borrador que había que verificar. Eso cambió nuestra forma de trabajar:

1. **Cada cifra del texto sale de un archivo.** Las tablas del blog las escriben los scripts (`resultados/`), y los números del texto se comparan con ellas. Así se encontraron A1, A6 y A8.
2. **Cada fórmula se prueba contra un caso conocido.** `pd_verificar.py` calcula las métricas a mano en cuatro créditos, comprueba la scorecard y reproduce los coeficientes de la logística con el método de Shapley.
3. **Cada referencia se abre.** Una referencia que no pudimos abrir o confirmar no se cita, aunque la IA la recordara con detalle (sección 10.2).
4. **Los errores se anotan cuando aparecen,** no al final. De ahí sale la sección 10.

La IA aceleró casi todo, pero no reemplazó la parte que más pesa en la nota: entender lo suficiente para darse cuenta de cuándo está equivocada.
