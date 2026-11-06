<!-- Sección 4. Autor: Sebastián (parte C).
     Numeración provisional: Tablas F1 y F2. construir_blog.py pone la final.
     Cada fórmula de esta sección se comprobó contra el código (riesgo/red.py, riesgo/metricas.py)
     y contra PyTorch 2.13.0 (torch/optim/adam.py y rmsprop.py); ver registro_ia.md. -->

## 4. Fundamentos matemáticos

Esta sección define todo lo que usan los modelos de las secciones 5 y 6: qué es una neurona y una red, de dónde salen las funciones de pérdida, cómo se calculan los gradientes, cómo se actualizan los pesos, cómo se regulariza y cómo se mide el resultado. La notación es la misma en todo el reporte: $x_i \in \mathbb{R}^d$ son las $d$ columnas de entrada del crédito $i$ después del preprocesamiento (89 en la PD y 77 en la LGD), $y_i$ es el valor observado (0 o 1 en la PD, una fracción entre 0 y 1 en la LGD) y $\hat{y}_i$ es lo que predice el modelo.

### 4.1 Neurona artificial

Una neurona artificial recibe un vector de entradas $x \in \mathbb{R}^d$, calcula una suma ponderada más un sesgo y le aplica una función de activación $\varphi$:

$$
z = w^\top x + b = \sum_{j=1}^{d} w_j x_j + b, \qquad a = \varphi(z)
$$

Los pesos $w \in \mathbb{R}^d$ y el sesgo $b \in \mathbb{R}$ son los parámetros que se aprenden; $z$ es la preactivación y $a$, la salida de la neurona. La idea viene del modelo de McCulloch y Pitts (1943), con salidas binarias, y del perceptrón de Rosenblatt (1958), que aprendía los pesos a partir de ejemplos.

Una sola neurona con activación sigmoide ya es la regresión logística de la sección 5: $\hat{p} = \sigma(w^\top x + b)$. Lo que agrega una red es poner neuronas en capas.

### 4.2 Perceptrón multicapa

Un perceptrón multicapa con $L$ capas ocultas encadena neuronas en capas. La capa $l$ tiene $n_l$ neuronas; cada una recibe todas las salidas de la capa anterior. Con $h^{(0)} = x$ y $n_0 = d$:

$$
z^{(l)} = W^{(l)} h^{(l-1)} + b^{(l)}, \qquad h^{(l)} = \varphi\left(z^{(l)}\right), \qquad l = 1, \dots, L
$$

donde $W^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}$ es la matriz de pesos (la fila $k$ son los pesos de la neurona $k$), $b^{(l)} \in \mathbb{R}^{n_l}$ son los sesgos y $\varphi$ se aplica a cada componente. La salida es una sola neurona con activación sigmoide:

$$
\hat{y} = \sigma\left(z^{(L+1)}\right), \qquad z^{(L+1)} = w^{(L+1)\top} h^{(L)} + b^{(L+1)}
$$

Es lo que hace `crear_red` en `riesgo/red.py`: `capas=[128, 64]` es $L = 2$, con $n_1 = 128$ y $n_2 = 64$. El número de parámetros entrenables es

$$
\sum_{l=1}^{L+1} (n_{l-1} + 1)\, n_l, \qquad n_{L+1} = 1
$$

Para la red de la PD (89 entradas, capas de 128 y 64) da $90 \cdot 128 + 129 \cdot 64 + 65 \cdot 1 = 19\,841$, el número que reporta la sección 5. Para la de la LGD (77 entradas, capas de 64 y 32), $78 \cdot 64 + 65 \cdot 32 + 33 = 7\,105$.

¿Por qué capas? Con una sola capa oculta, una activación de tipo sigmoide y suficientes neuronas, un perceptrón multicapa puede aproximar cualquier función continua sobre un conjunto compacto tan bien como se quiera (Hornik et al., 1989). Eso dice que la red *puede* representar relaciones no lineales entre las variables; no dice que las vaya a aprender con los datos que tenemos, ni que esas relaciones existan. Las secciones 5 y 6 miden cuánto ganamos con eso en la práctica.

### 4.3 Funciones de activación

La Tabla F1 resume las tres activaciones que usamos y sus derivadas, que son las que necesita la retropropagación (sección 4.5).

**Tabla F1.** Funciones de activación usadas, sus derivadas y dónde aparecen.

| Activación | $\varphi(z)$ | $\varphi'(z)$ | Rango | Dónde se usa |
|:--|:--|:--|:--|:--|
| ReLU | $\max(0, z)$ | $1$ si $z > 0$; $0$ si $z < 0$ | $[0, \infty)$ | Capas ocultas de la red de la PD |
| Tangente hiperbólica | $\tanh z = \dfrac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$ | $1 - \tanh^2 z$ | $(-1, 1)$ | Capas ocultas de la red de la LGD |
| Sigmoide | $\sigma(z) = \dfrac{1}{1 + e^{-z}}$ | $\sigma(z)\,(1 - \sigma(z))$ | $(0, 1)$ | Neurona de salida de las dos redes |

*Fuente: elaboración propia. Las activaciones de cada red salen de la búsqueda de hiperparámetros (secciones 5.2 y 6.3).*

Algunos detalles que importan:

- **ReLU** (Nair y Hinton, 2010) no es derivable en $z = 0$. PyTorch usa 0 como derivada en ese punto; en la práctica casi nunca se cae exactamente en 0. Su derivada es 1 para toda $z > 0$, así que el gradiente no se encoge al pasar por muchas capas, como sí pasa con la sigmoide.
- **tanh** es una sigmoide reescalada: $\tanh z = 2\sigma(2z) - 1$. Está centrada en cero, pero se satura igual: para $|z|$ grande, $1 - \tanh^2 z \approx 0$.
- **Sigmoide.** La derivada sale de derivar $(1 + e^{-z})^{-1}$:

$$
\sigma'(z) = \frac{e^{-z}}{(1 + e^{-z})^2} = \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}} = \sigma(z)\,\bigl(1 - \sigma(z)\bigr)
$$

porque $\dfrac{e^{-z}}{1 + e^{-z}} = 1 - \dfrac{1}{1 + e^{-z}}$. La usamos en la salida porque las dos cosas que estimamos están entre 0 y 1: la PD es una probabilidad y la LGD es una fracción. Como $\sigma(z) \in (0, 1)$ para todo $z$, la red nunca predice algo imposible, como una PD negativa o una LGD de 1.3. La sección 6 muestra que la regresión lineal sí puede salirse de ese rango.

### 4.4 Funciones de pérdida por máxima verosimilitud

Una función de pérdida no se escoge al gusto: cada una corresponde a un supuesto sobre cómo se generan los datos. Si suponemos un modelo probabilístico para $y_i$ dado $x_i$, con parámetros $\theta$, el principio de máxima verosimilitud pide los $\theta$ que hacen más probable lo observado. Con observaciones independientes, la verosimilitud es un producto, y es más cómodo maximizar su logaritmo:

$$
\hat{\theta} = \arg\max_\theta \prod_{i=1}^{n} f(y_i \mid x_i; \theta) = \arg\min_\theta \left[ -\frac{1}{n} \sum_{i=1}^{n} \ln f(y_i \mid x_i; \theta) \right]
$$

El logaritmo no cambia dónde está el máximo porque es creciente, y dividir por $n$ tampoco. Lo que queda entre corchetes, el negativo del logaritmo de la verosimilitud promediado, es la función de pérdida que se minimiza.

**Entropía cruzada binaria, desde Bernoulli (PD).** Si $y_i \in \{0, 1\}$ y el modelo dice que la probabilidad de ser malo es $\hat{p}_i$, entonces $y_i \sim \text{Bernoulli}(\hat{p}_i)$:

$$
f(y_i \mid x_i; \theta) = \hat{p}_i^{\,y_i} (1 - \hat{p}_i)^{1 - y_i}
$$

que vale $\hat{p}_i$ si $y_i = 1$ y $1 - \hat{p}_i$ si $y_i = 0$. Su logaritmo es $y_i \ln \hat{p}_i + (1 - y_i) \ln(1 - \hat{p}_i)$, y con el signo menos y el promedio queda la entropía cruzada binaria:

$$
\mathcal{L}_{\text{BCE}} = -\frac{1}{n} \sum_{i=1}^{n} \Bigl[ y_i \ln \hat{p}_i + (1 - y_i) \ln (1 - \hat{p}_i) \Bigr]
$$

Es lo que calcula `nn.BCELoss()` de PyTorch, que por defecto promedia sobre el lote. Su derivada respecto a la preactivación de salida es muy simple. Por la regla de la cadena, con $\hat{p}_i = \sigma(z_i)$, para un crédito:

$$
\frac{\partial \ell_i}{\partial \hat{p}_i} = -\frac{y_i}{\hat{p}_i} + \frac{1 - y_i}{1 - \hat{p}_i} = \frac{\hat{p}_i - y_i}{\hat{p}_i (1 - \hat{p}_i)}, \qquad \frac{\partial \ell_i}{\partial z_i} = \frac{\hat{p}_i - y_i}{\hat{p}_i (1 - \hat{p}_i)} \cdot \hat{p}_i (1 - \hat{p}_i) = \hat{p}_i - y_i
$$

donde $\ell_i = -[y_i \ln \hat{p}_i + (1 - y_i) \ln (1 - \hat{p}_i)]$. El factor $\hat{p}_i(1 - \hat{p}_i)$ de la sigmoide se cancela. Por eso la sigmoide y la entropía cruzada van juntas: el gradiente es el error de predicción, sin ningún factor que lo apague cuando la sigmoide se satura.

**Entropía cruzada con objetivo fraccional (LGD).** En la LGD $y_i \in [0, 1]$ no es 0 o 1, así que la Bernoulli no es una distribución posible para $y_i$ y $\mathcal{L}_{\text{BCE}}$ ya no es una verosimilitud. Es una *cuasi-verosimilitud* (Papke y Wooldridge, 1996): se usa la misma fórmula y lo único que se supone es que el modelo para la media, $\mathbb{E}[y \mid x] = \hat{y}$, es correcto. La sección 6.2 muestra por qué basta con eso: la pérdida esperada se minimiza justo en $\hat{y} = \mathbb{E}[y \mid x]$.

**Error cuadrático, desde Gauss.** Si suponemos $y_i = \hat{y}_i + \varepsilon_i$ con $\varepsilon_i \sim \mathcal{N}(0, s^2)$ independientes y $s$ fija, entonces

$$
f(y_i \mid x_i; \theta) = \frac{1}{\sqrt{2\pi s^2}} \exp\left( -\frac{(y_i - \hat{y}_i)^2}{2 s^2} \right)
$$

y el negativo del logaritmo de la verosimilitud es

$$
-\sum_{i=1}^{n} \ln f(y_i \mid x_i; \theta) = \frac{n}{2} \ln (2\pi s^2) + \frac{1}{2 s^2} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

El primer término no depende de $\theta$ y $1/(2s^2)$ es una constante positiva, así que minimizarlo es lo mismo que minimizar el error cuadrático medio:

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Es lo que optimiza la regresión lineal de la LGD (mínimos cuadrados) y la red de la LGD con `--set lgd.perdida=mse`. Con una salida sigmoide, su derivada respecto a $z_i$ es $\frac{2}{n} (\hat{y}_i - y_i)\, \hat{y}_i (1 - \hat{y}_i)$ (el $\frac{1}{n}$ por el promedio y el 2 por el cuadrado). Aquí el factor de la sigmoide no se cancela: cerca de 0 o de 1 la red aprende muy despacio, y en la LGD el 45 % de los créditos de prueba tiene LGD = 1 (sección 6). Además, la Gaussiana supone que $y$ puede tomar cualquier valor real, y la LGD no. Por las dos razones usamos la entropía cruzada por defecto y dejamos el error cuadrático para comparar.

### 4.5 Retropropagación

Para minimizar la pérdida hace falta su gradiente respecto a cada peso. La retropropagación (Rumelhart et al., 1986) lo calcula aplicando la regla de la cadena de la salida hacia la entrada, capa por capa, y reutilizando lo que ya calculó.

Para un crédito, se define el error de la capa $l$ como la derivada de la pérdida respecto a su preactivación, $\delta^{(l)} = \partial \ell / \partial z^{(l)} \in \mathbb{R}^{n_l}$. El algoritmo tiene cuatro pasos:

1. **Hacia adelante.** Calcular y guardar $z^{(l)}$ y $h^{(l)}$ de todas las capas con las ecuaciones de la sección 4.2.
2. **Error de salida.** Con sigmoide y entropía cruzada, por la sección 4.4: $\delta^{(L+1)} = \hat{y} - y$.
3. **Hacia atrás.** Para $l = L, \dots, 1$:

$$
\delta^{(l)} = \left( W^{(l+1)\top} \delta^{(l+1)} \right) \odot \varphi'\left(z^{(l)}\right)
$$

   donde $\odot$ es el producto componente a componente. Sale de la regla de la cadena: $z^{(l)}_k$ afecta la pérdida a través de todas las neuronas $j$ de la capa siguiente, con $\partial z^{(l+1)}_j / \partial z^{(l)}_k = W^{(l+1)}_{jk}\, \varphi'(z^{(l)}_k)$, y por eso $\delta^{(l)}_k = \sum_j \delta^{(l+1)}_j W^{(l+1)}_{jk}\, \varphi'(z^{(l)}_k)$. La suma sobre $j$ es lo que escribe la matriz transpuesta.

4. **Gradientes.** Como $z^{(l)}_j = \sum_k W^{(l)}_{jk} h^{(l-1)}_k + b^{(l)}_j$:

$$
\frac{\partial \ell}{\partial W^{(l)}} = \delta^{(l)} \, h^{(l-1)\top}, \qquad \frac{\partial \ell}{\partial b^{(l)}} = \delta^{(l)}
$$

Con un lote de $m$ créditos y la pérdida promediada, los gradientes son el promedio de los de cada crédito. En el código todo esto lo hace `perdida_lote.backward()`, que aplica estas mismas reglas sobre el grafo de operaciones que PyTorch guardó en el paso hacia adelante; `optimizador.step()` usa después esos gradientes para actualizar los pesos (sección 4.6).

### 4.6 Optimizador: Adam

El descenso de gradiente más simple actualiza $\theta \leftarrow \theta - \alpha\, g$, con $g$ el gradiente del lote y $\alpha$ la tasa de aprendizaje. Adam (Kingma y Ba, 2015) le agrega dos promedios móviles por parámetro: uno del gradiente (la dirección) y otro del gradiente al cuadrado (el tamaño típico). En el paso $t$, con $m_0 = v_0 = 0$:

$$
\begin{aligned}
g_t &= \nabla_\theta \mathcal{L}(\theta_{t-1}) + \lambda\, \theta_{t-1} \\
m_t &= \beta_1 m_{t-1} + (1 - \beta_1)\, g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\
\theta_t &= \theta_{t-1} - \alpha\, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
$$

Las operaciones son componente a componente. Usamos los valores por defecto de PyTorch, $\beta_1 = 0.9$, $\beta_2 = 0.999$ y $\epsilon = 10^{-8}$, y la tasa $\alpha$ que eligió la búsqueda ($3 \cdot 10^{-4}$ en la PD y $10^{-3}$ en la LGD). El término $\lambda\theta_{t-1}$ es la penalización $L_2$ (sección 4.7).

Por qué cada pieza:

- **$m_t$** suaviza el gradiente de lote a lote, como un momento: las direcciones que se repiten se refuerzan y el ruido se cancela.
- **$\sqrt{\hat{v}_t}$** divide cada componente por su tamaño típico, así que cada parámetro tiene, en la práctica, su propia tasa de aprendizaje. El paso queda del orden de $\alpha$ aunque el gradiente sea muy grande o muy pequeño.
- **La corrección de sesgo** $1 - \beta^t$ existe porque $m_0 = v_0 = 0$: en los primeros pasos los promedios están sesgados hacia cero. Por ejemplo, en $t = 1$, $m_1 = (1 - \beta_1) g_1 = 0.1\, g_1$, y al dividir por $1 - \beta_1 = 0.1$ se recupera $\hat{m}_1 = g_1$.

Comprobamos estas ecuaciones contra el código instalado de PyTorch 2.13.0 (`torch/optim/adam.py`): calcula el paso como $\frac{\alpha}{1 - \beta_1^t} \cdot \frac{m_t}{\sqrt{v_t}/\sqrt{1 - \beta_2^t} + \epsilon}$, que es la misma expresión.

La búsqueda de la PD también probó RMSprop (Hinton et al., 2012), que es Adam sin el promedio del gradiente ni la corrección de sesgo: $v_t = \rho\, v_{t-1} + (1 - \rho)\, g_t^2$ y $\theta_t = \theta_{t-1} - \alpha\, g_t / (\sqrt{v_t} + \epsilon)$. Las diapositivas originales usan $\rho = 0.9$; PyTorch usa $\rho = 0.99$ por defecto, y ese es el valor con que corrió nuestra búsqueda.

### 4.7 Regularización

Una red con miles de parámetros puede memorizar el ruido del entrenamiento. Usamos tres técnicas para evitarlo, y sus intensidades son hiperparámetros de la búsqueda.

**Penalización $L_2$.** Se le suma a la pérdida el cuadrado de la norma de los parámetros:

$$
J(\theta) = \mathcal{L}(\theta) + \frac{\lambda}{2} \lVert \theta \rVert_2^2, \qquad \nabla_\theta J = \nabla_\theta \mathcal{L} + \lambda\, \theta
$$

Así, los pesos grandes cuestan, y la red prefiere soluciones más suaves. En el código es `weight_decay=λ` de PyTorch, que suma exactamente $\lambda\theta$ al gradiente antes de entrar a Adam (la primera línea de las ecuaciones de la sección 4.6). Hay dos detalles. Primero, se aplica a todos los parámetros, sesgos incluidos. Segundo, como el término pasa por la división entre $\sqrt{\hat{v}_t}$, en Adam no equivale a encoger cada peso en la misma proporción. Esa otra versión, el *weight decay desacoplado* de AdamW (Loshchilov y Hutter, 2019), no es la que usamos. La regresión logística también lleva $L_2$, con $C$ como el inverso de su fuerza (sección 5.1).

**Dropout.** En cada paso de entrenamiento, cada neurona oculta se apaga con probabilidad $q$, independiente de las demás (Srivastava et al., 2014). Con $r_k \sim \text{Bernoulli}(1 - q)$:

$$
\tilde{h}^{(l)}_k = \frac{r_k}{1 - q}\, h^{(l)}_k
$$

Dividir por $1 - q$ hace que el valor esperado no cambie: $\mathbb{E}[\tilde{h}_k] = \frac{(1 - q)\, h_k}{1 - q} = h_k$. Por eso, al predecir, el dropout se apaga y no hay que corregir nada. Es lo que hace `nn.Dropout(q)` de PyTorch entre `red.train()` y `red.eval()`. (El artículo original no divide al entrenar, sino que multiplica los pesos por $1 - q$ al predecir. En valor esperado es lo mismo.) La idea es que ninguna neurona puede depender de que otra esté presente. En la retropropagación, el gradiente pasa solo por las neuronas que quedaron encendidas, multiplicado por el mismo $r_k / (1 - q)$. La red de la PD usa $q = 0.3$ y la de la LGD, $q = 0$.

**Parada temprana.** Después de cada época se mide la pérdida en validación, $\mathcal{L}_{\text{val}}(\theta_e)$. Se guarda la mejor época,

$$
e^* = \arg\min_{e' \le e} \mathcal{L}_{\text{val}}(\theta_{e'})
$$

y el entrenamiento se detiene en la primera época $e$ con $e - e^* \ge P$, donde $P$ es la paciencia ($P = 6$ en `config.yaml`), o al llegar al máximo de 60 épocas. Al final se recuperan los pesos $\theta_{e^*}$. Regulariza porque la red deja de ajustarse justo cuando empieza a mejorar en entrenamiento y a empeorar en datos que no ve. Hay muchos criterios para decidir cuándo parar (Prechelt, 1998); el nuestro, la paciencia, es uno de los más simples. La decisión la toma la validación, nunca la prueba.

### 4.8 Métricas de evaluación

La Tabla F2 resume las métricas y qué miden. Las de clasificación están en `riesgo/metricas.py` y se comprobaron a mano en un caso pequeño (`python pd_verificar.py`).

**Tabla F2.** Métricas de evaluación usadas en el reporte.

| Métrica | Tarea | Qué mide | Rango | Mejor |
|:--|:--|:--|:--|:--|
| AUC | PD | Discriminación: si los malos tienen PD más alta que los buenos | $[0, 1]$, 0.5 es azar | Más alto |
| Gini | PD | Lo mismo que el AUC, en otra escala | $[-1, 1]$, 0 es azar | Más alto |
| KS | PD | La mayor separación entre las PD de malos y de buenos | $[0, 1]$ | Más alto |
| Brier | PD | Calibración y discriminación juntas: el error cuadrático de la probabilidad | $[0, 1]$ | Más bajo |
| MAE | LGD | Error típico en unidades de LGD | $[0, \infty)$ | Más bajo |
| RMSE | LGD | Error típico, castigando más los errores grandes | $[0, \infty)$ | Más bajo |

*Fuente: elaboración propia.*

**Curva ROC y AUC.** Para un umbral $t$, se predice "malo" si $\hat{p}_i > t$. La tasa de verdaderos positivos es la fracción de malos que el umbral atrapa y la de falsos positivos, la fracción de buenos que rechaza sin razón:

$$
\text{TPR}(t) = \frac{\#\{i : y_i = 1,\ \hat{p}_i > t\}}{n_1}, \qquad \text{FPR}(t) = \frac{\#\{i : y_i = 0,\ \hat{p}_i > t\}}{n_0}
$$

con $n_1$ malos y $n_0$ buenos. La curva ROC une los puntos $(\text{FPR}(t), \text{TPR}(t))$ al mover $t$, y el AUC es el área bajo ella (Fawcett, 2006). Tiene una lectura más directa: es la probabilidad de que un malo tomado al azar tenga una PD más alta que un bueno tomado al azar, con los empates valiendo la mitad (Hanley y McNeil, 1982):

$$
\text{AUC} = \frac{1}{n_1 n_0} \sum_{i : y_i = 1} \sum_{j : y_j = 0} \left( \mathbb{1}[\hat{p}_i > \hat{p}_j] + \tfrac{1}{2}\, \mathbb{1}[\hat{p}_i = \hat{p}_j] \right)
$$

Solo depende del orden de las PD, no de su valor. Por eso no cambia con la escala de la scorecard (sección 5.5).

**Índice de Gini.** $\text{Gini} = 2\,\text{AUC} - 1$. Es el AUC reescalado para que el azar sea 0 y la separación perfecta, 1.

**Estadístico KS.** Es la mayor distancia vertical entre las distribuciones acumuladas de la PD de malos y de buenos, que en un umbral $t$ son $1 - \text{TPR}(t)$ y $1 - \text{FPR}(t)$:

$$
\text{KS} = \max_t \bigl| \text{TPR}(t) - \text{FPR}(t) \bigr|
$$

Es el estadístico de la prueba de Kolmogorov-Smirnov de dos muestras (Massey, 1951) aplicado a las PD de los dos grupos. En la curva ROC es la mayor distancia vertical entre la curva y la diagonal.

**Puntaje de Brier.** El error cuadrático medio de la probabilidad, con $y_i \in \{0, 1\}$ (Brier, 1950):

$$
\text{Brier} = \frac{1}{n} \sum_{i=1}^{n} (\hat{p}_i - y_i)^2
$$

A diferencia de las anteriores, sí depende del valor de las PD: un modelo que ordena bien, pero dice 40 % donde la tasa real es 20 %, tiene buen AUC y mal Brier. Como referencia, predecir para todos la tasa de malos $\pi$ de la partición da $\text{Brier} = \pi(1 - \pi)$ (sección 5.4).

**Curva de calibración.** Se ordenan los créditos por PD predicha, se parten en diez grupos del mismo tamaño (deciles) y en cada uno se compara la PD media predicha con la tasa de malos observada. Si el modelo está bien calibrado, los puntos caen sobre la diagonal.

**MAE y RMSE (LGD).**

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \lvert y_i - \hat{y}_i \rvert, \qquad \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

Los dos están en unidades de LGD: un MAE de 0.076 quiere decir que, en promedio, la predicción se equivoca en 7.6 puntos porcentuales de la exposición. El RMSE es siempre mayor o igual que el MAE y lo es más cuando hay errores grandes, porque los eleva al cuadrado.

### 4.9 Scorecard

La derivación de la transformación de probabilidad a puntaje (puntaje base, odds de referencia y PDO), y cómo se reparten los puntos entre las variables, están en la sección 5.5, junto a los resultados que las usan.

#### Referencias citadas en esta sección

Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review, 78*(1), 1–3. https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2

Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters, 27*(8), 861–874. https://doi.org/10.1016/j.patrec.2005.10.010

Hanley, J. A., y McNeil, B. J. (1982). The meaning and use of the area under a receiver operating characteristic (ROC) curve. *Radiology, 143*(1), 29–36. https://doi.org/10.1148/radiology.143.1.7063747

Hinton, G., Srivastava, N., y Swersky, K. (2012). *Neural networks for machine learning. Lecture 6e: rmsprop: Divide the gradient by a running average of its recent magnitude* [Diapositivas]. University of Toronto. https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

Hornik, K., Stinchcombe, M., y White, H. (1989). Multilayer feedforward networks are universal approximators. *Neural Networks, 2*(5), 359–366. https://doi.org/10.1016/0893-6080(89)90020-8

Kingma, D. P., y Ba, J. (2015). *Adam: A method for stochastic optimization* [Ponencia]. 3rd International Conference on Learning Representations, San Diego, CA, Estados Unidos. https://arxiv.org/abs/1412.6980

Loshchilov, I., y Hutter, F. (2019). *Decoupled weight decay regularization* [Ponencia]. 7th International Conference on Learning Representations, Nueva Orleans, LA, Estados Unidos. https://arxiv.org/abs/1711.05101

Massey, F. J. (1951). The Kolmogorov-Smirnov test for goodness of fit. *Journal of the American Statistical Association, 46*(253), 68–78. https://doi.org/10.1080/01621459.1951.10500769

McCulloch, W. S., y Pitts, W. (1943). A logical calculus of the ideas immanent in nervous activity. *The Bulletin of Mathematical Biophysics, 5*(4), 115–133. https://doi.org/10.1007/BF02478259

Nair, V., y Hinton, G. E. (2010). Rectified linear units improve restricted Boltzmann machines. En *Proceedings of the 27th International Conference on Machine Learning*. https://icml.cc/Conferences/2010/papers/432.pdf

Papke, L. E., y Wooldridge, J. M. (1996). Econometric methods for fractional response variables with an application to 401(k) plan participation rates. *Journal of Applied Econometrics, 11*(6), 619–632. https://doi.org/10.1002/(SICI)1099-1255(199611)11:6<619::AID-JAE418>3.0.CO;2-1

Prechelt, L. (1998). Early stopping - but when? En G. B. Orr y K.-R. Müller (Eds.), *Neural networks: Tricks of the trade* (pp. 55–69). Springer. https://doi.org/10.1007/3-540-49430-8_3

Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. *Psychological Review, 65*(6), 386–408. https://doi.org/10.1037/h0042519

Rumelhart, D. E., Hinton, G. E., y Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature, 323*(6088), 533–536. https://doi.org/10.1038/323533a0

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., y Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research, 15*(56), 1929–1958. https://jmlr.org/papers/v15/srivastava14a.html
