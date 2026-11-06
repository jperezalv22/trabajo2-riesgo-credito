# Guion del video promocional: "¿Me prestas?"

Duración objetivo: **2 min 40 s** (el enunciado pide de 2 a 3 minutos). Aparecen los tres.
Grabación sugerida: celular en horizontal, buena luz de frente, y la pantalla de la app grabada aparte
(OBS o el grabador de Windows) para intercalarla. Se sube a YouTube como **no listado** y el enlace va en
`config.yaml` (`app.enlaces.video`): de ahí lo toman la app y el blog (`python construir_blog.py`).

| Tiempo | Imagen | Quién | Qué se dice |
|---|---|---|---|
| 0:00-0:15 | Primer plano. Sebastián con el celular en la mano, cara de preocupación. | Sebastián | "Llevo semanas pensando en pedir un préstamo. Y la pregunta que no me deja dormir no es cuánto voy a pagar. Es esta: ¿me lo van a aprobar?" |
| 0:15-0:35 | Juan Pablo frente al tablero o el computador. | Juan Pablo | "Antes de prestarle a alguien, un banco se hace dos preguntas: qué tan probable es que no le paguen y, si no le pagan, cuánto pierde. Las responde con modelos que la persona nunca ve. Uno pide el crédito a ciegas." |
| 0:35-0:45 | Los tres juntos, de frente. | Los tres | "¿Y si usted pudiera ver lo que ve el banco?" (Santiago:) "Por eso hicimos *¿Me prestas?*" |
| 0:45-1:25 | Grabación de pantalla de la app. Voz en off de Santiago, con su cara en una esquina. | Santiago | "Usted entra, y sus datos ya vienen llenos con los de una persona típica: solo cambia lo que es distinto en su caso. Su ingreso, el monto, el plazo. Al instante ve su puntaje, su probabilidad de no pagar y dónde queda frente a más de cien mil personas. Y no es una caja negra: le mostramos qué le suma puntos y qué le resta." |
| 1:25-1:55 | Pantalla: sección "¿Qué pasaría si...?". Se cambia el plazo de 36 a 60 meses y se ve la caída del puntaje. | Sebastián | "Y lo mejor: puede probar antes de pedir. ¿Qué pasa si pido el préstamo a 60 meses en vez de 36? Pierdo casi 18 puntos. ¿Y si pido menos plata? Ahora usted decide con información, no a ciegas." |
| 1:55-2:20 | Juan Pablo y Santiago. Intercalar la Figura de la ROC y la tabla de las 30 semillas del blog. | Juan Pablo / Santiago | JP: "Detrás hay una red neuronal entrenada con 130 mil créditos reales de Lending Club." Santiago: "La comparamos de forma justa con un modelo clásico, con 30 semillas distintas. Y le contamos todo, incluso dónde no ganó." |
| 2:20-2:40 | Los tres de frente. En pantalla, el enlace de la app y del blog. | Los tres | Sebastián: "Pruébela, es gratis." Juan Pablo: "El enlace está aquí abajo." Santiago: "*¿Me prestas?* Antes de que el banco decida, decida usted." |

## Notas para la grabación

- Los números que se dicen salen del trabajo y hay que decirlos así: **130 832 créditos** de entrenamiento
  (se puede decir "130 mil"), "más de cien mil personas" en el histograma (la población de entrenamiento
  de `modelos/pd_puntajes.npy`), **30 semillas**, y "casi 18 puntos" al pasar a 60 meses con la persona
  típica (−17.8 en la app). Si alguien cambia los datos de ejemplo, hay que revisar esta última cifra en
  la app antes de grabar.
- No decir que la app "le dice si lo aprueban": muestra lo que estima un modelo académico, entrenado con
  datos de Estados Unidos de 2007 a 2013. La frase final del video lo deja implícito, y la app lo dice
  al final de la página.
- Tampoco decir que la red "es mucho mejor": le gana a la logística por 0.008 de AUC (sección 5). El guion
  dice "le contamos todo, incluso dónde no ganó", que es verdad y vende la honestidad del trabajo.
- Antes de grabar la pantalla, confirmar que la app publicada carga y que el enlace del blog funciona.
