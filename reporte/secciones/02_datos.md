<!-- Sección 2. Autor: Juan Pablo Pérez Álvarez.
     Rutas de figuras relativas a reporte/, como en el Trabajo 1.
     La numeración de figuras y tablas es provisional: se ajusta al unir el blog.
     Todas las tablas salen de `python descriptivo.py` (resultados/descriptivo/). -->

## 2. Datos

### 2.1 De dónde salen

Usamos el conjunto *Credit Risk Analysis* publicado en Kaggle (ranadeep, 2021), que reúne los créditos de consumo otorgados por la plataforma estadounidense Lending Club. Lo descargamos el 26 de octubre de 2026. El archivo `loan.csv` tiene 887 379 créditos y 74 variables. Los créditos se originaron entre junio de 2007 y diciembre de 2015, y el último pago registrado es de enero de 2016, así que lo que sabemos de cada crédito es lo que había pasado hasta ese mes.

El conjunto trae un diccionario de variables (`LCDataDictionary.xlsx`, hoja *LoanStats*), que copiamos en `datos/diccionario/` del repositorio. Las definiciones de este reporte salen de ese diccionario y en la Tabla 2 están citadas textualmente, en inglés, como aparecen allí. Solo un nombre no coincide: la columna `verification_status` del archivo aparece en el diccionario como `is_inc_v`.

El archivo original pesa 442 MB y no cabe en el repositorio. Por eso guardamos las tablas ya procesadas (`datos/procesados/pd.parquet` y `lgd.parquet`) junto con el código que las genera desde el original (`python preparar_datos.py`, unos 20 segundos).

### 2.2 Variable objetivo

La variable objetivo `malo` sale de `loan_status` con la codificación del enunciado: 1 si el crédito se castigó, está en mora de más de 30 días o en *Default*; 0 si se pagó completo. Los estados que todavía no tienen un desenlace (*Current*, *Issued*, *In Grace Period* y *Late (16-30 days)*) quedan como NA y no se usan. La Tabla 1 muestra cuántos créditos hay en cada estado.

**Tabla 1.** Estados del crédito (`loan_status`) y su codificación.

| loan_status                                         | Codificación   | Créditos   |
|:----------------------------------------------------|:---------------|:-----------|
| Current                                             | NA             | 601 779    |
| Fully Paid                                          | 0              | 207 723    |
| Charged Off                                         | 1              | 45 248     |
| Late (31-120 days)                                  | 1              | 11 591     |
| Issued                                              | NA             | 8 460      |
| In Grace Period                                     | NA             | 6 253      |
| Late (16-30 days)                                   | NA             | 2 357      |
| Does not meet the credit policy. Status:Fully Paid  | 0              | 1 988      |
| Default                                             | 1              | 1 219      |
| Does not meet the credit policy. Status:Charged Off | 1              | 761        |

*Fuente: elaboración propia con datos de Lending Club (ranadeep, 2021) y la codificación del enunciado.*

Como se ve en la Tabla 1, la mayoría de los créditos todavía se estaba pagando en enero de 2016: 601 779 están en *Current*. En total, 618 849 créditos (69.7 %) quedan como NA. Con desenlace conocido quedan 268 530, de los cuales 58 819 son malos, una tasa de 21.9 %.

### 2.3 Variables de entrada

El modelo se usa cuando una persona pide el crédito, antes de aprobarlo (sección 1). Por eso solo entran variables que se conocen en ese momento: lo que la persona declara en la solicitud y lo que dice su reporte de crédito. La Tabla 2 muestra las 24 variables de entrada (19 numéricas y 5 categóricas), su definición en el diccionario, la transformación que les hicimos y el porcentaje de datos faltantes.

Dejamos por fuera cuatro grupos de variables:

- **Lo que pasa después del préstamo**, como `total_pymnt`, `recoveries`, `last_pymnt_d` u `out_prncp`. En el momento de decidir todavía no existen, y usarlas sería darle al modelo información del futuro.
- **`grade`, `sub_grade`, `int_rate` e `installment`.** El diccionario define `grade` como *"LC assigned loan grade"*: es la calificación que Lending Club le pone a cada crédito con su propio modelo de riesgo, y la tasa de interés y la cuota dependen de ella. Usarlas sería copiar el modelo de Lending Club en vez de construir el nuestro. Las dejamos en las tablas procesadas para quien quiera hacer esa comparación.
- **`tot_cur_bal`, `tot_coll_amt` y `total_rev_hi_lim`.** Están vacías en todos los créditos originados hasta 2011, en el 52 % de los de 2012 y completas desde 2013. Si las usáramos, el modelo podría aprender el año del crédito solo con mirar si están vacías.
- **Los campos de solicitudes conjuntas (`*_joint`, vacíos en el 99.9 %) y los `open_*` e `il_*` (vacíos en el 97.6 %).**

A las variables que quedaron les hicimos estas transformaciones:

- `term` (*"36 months"* o *"60 months"*) pasa a número de meses, y `emp_length` pasa a años, con *"< 1 year"* como 0 y *"10+ years"* como 10, igual que lo describe el diccionario.
- `earliest_cr_line` es una fecha; la convertimos en la antigüedad de la historia crediticia en meses al momento del préstamo.
- En `mths_since_last_delinq` y `mths_since_last_record` un vacío no es un dato perdido: quiere decir que la persona nunca ha estado en mora o nunca ha tenido un registro público. Por eso, antes de imputar, guardamos esa información en dos indicadores, `nunca_en_mora` y `sin_registros_publicos`. Así el modelo no confunde "nunca" con "el valor mediano".
- El ingreso, el monto del préstamo y el saldo revolvente tienen colas muy largas a la derecha; les aplicamos $\log(1 + x)$.
- Las numéricas se imputan con la mediana y se estandarizan. Las categóricas se codifican en *one-hot* y las categorías con menos de 50 créditos se agrupan en una sola.

Todo esto está en `riesgo/preparar.py` y se ajusta **solo con el conjunto de entrenamiento**: la mediana, la media y la desviación de cada variable salen de ahí y se aplican sin cambios a validación y prueba. Todos los modelos de una misma tarea usan el mismo preprocesamiento.

**Tabla 2.** Variables de entrada, definición en el diccionario de Lending Club, transformación y porcentaje de faltantes.

| Variable                   | Columna original           | Definición en el diccionario                                                                                                                                                                             | Transformación                             | Faltantes   |
|:---------------------------|:---------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------|:------------|
| loan_amnt                  | loan_amnt                  | The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.                             | log(1 + x)                                 | 0.0%        |
| plazo_meses                | term                       | The number of payments on the loan. Values are in months and can be either 36 or 60.                                                                                                                     | "36 months" -> 36                          | 0.0%        |
| antiguedad_empleo          | emp_length                 | Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.                                                                        | "< 1 year" -> 0, "10+ years" -> 10         | 4.0%        |
| annual_inc                 | annual_inc                 | The self-reported annual income provided by the borrower during registration.                                                                                                                            | log(1 + x)                                 | 0.0%        |
| dti                        | dti                        | A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income. |                                            | 0.0%        |
| delinq_2yrs                | delinq_2yrs                | The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years                                                                                             |                                            | 0.0%        |
| antiguedad_credito_meses   | earliest_cr_line           | The month the borrower's earliest reported credit line was opened                                                                                                                                        | meses entre esa fecha y la originación     | 0.0%        |
| inq_last_6mths             | inq_last_6mths             | The number of inquiries in past 6 months (excluding auto and mortgage inquiries)                                                                                                                         |                                            | 0.0%        |
| meses_desde_mora           | mths_since_last_delinq     | The number of months since the borrower's last delinquency.                                                                                                                                              | vacío = nunca (ver nunca_en_mora)          | 55.2%       |
| meses_desde_registro       | mths_since_last_record     | The number of months since the last public record.                                                                                                                                                       | vacío = nunca (ver sin_registros_publicos) | 87.1%       |
| nunca_en_mora              | mths_since_last_delinq     | The number of months since the borrower's last delinquency.                                                                                                                                              | 1 si está vacío, 0 si no                   | 0.0%        |
| sin_registros_publicos     | mths_since_last_record     | The number of months since the last public record.                                                                                                                                                       | 1 si está vacío, 0 si no                   | 0.0%        |
| open_acc                   | open_acc                   | The number of open credit lines in the borrower's credit file.                                                                                                                                           |                                            | 0.0%        |
| pub_rec                    | pub_rec                    | Number of derogatory public records                                                                                                                                                                      |                                            | 0.0%        |
| revol_bal                  | revol_bal                  | Total credit revolving balance                                                                                                                                                                           | log(1 + x)                                 | 0.0%        |
| revol_util                 | revol_util                 | Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.                                                                               |                                            | 0.1%        |
| total_acc                  | total_acc                  | The total number of credit lines currently in the borrower's credit file                                                                                                                                 |                                            | 0.0%        |
| collections_12_mths_ex_med | collections_12_mths_ex_med | Number of collections in 12 months excluding medical collections                                                                                                                                         |                                            | 0.1%        |
| acc_now_delinq             | acc_now_delinq             | The number of accounts on which the borrower is now delinquent.                                                                                                                                          |                                            | 0.0%        |
| home_ownership             | home_ownership             | The home ownership status provided by the borrower during registration. Our values are: RENT, OWN, MORTGAGE, OTHER.                                                                                      |                                            | 0.0%        |
| verification_status        | verification_status        | Indicates if income was verified by LC, not verified, or if the income source was verified                                                                                                               | en el diccionario se llama is_inc_v        | 0.0%        |
| purpose                    | purpose                    | A category provided by the borrower for the loan request.                                                                                                                                                |                                            | 0.0%        |
| addr_state                 | addr_state                 | The state provided by the borrower in the loan application                                                                                                                                               |                                            | 0.0%        |
| initial_list_status        | initial_list_status        | The initial listing status of the loan. Possible values are – W, F                                                                                                                                       |                                            | 0.0%        |

*Fuente: elaboración propia; definiciones copiadas de `LCDataDictionary.xlsx`, hoja LoanStats (ranadeep, 2021). Faltantes calculados sobre los 268 530 créditos con desenlace conocido.*

### 2.4 Particiones de entrenamiento, validación y prueba

En la práctica el modelo se entrena con créditos del pasado y se usa con clientes nuevos. Para medir eso, la prueba es temporal: entrenamos y validamos con los créditos originados hasta 2013, y probamos con los originados en 2014 y 2015. Dentro de los años de entrenamiento, el 20 % va a validación, elegido al azar con la semilla 2026 y estratificado por la variable objetivo para que las dos partes tengan la misma tasa de malos. La validación se usa para la parada temprana y para elegir hiperparámetros; la prueba solo se mira al final. La Tabla 3 resume las tres particiones.

**Tabla 3.** Particiones para la probabilidad de incumplimiento.

| Partición     | Créditos   | Años      | Tasa de malos   |
|:--------------|:-----------|:----------|:----------------|
| entrenamiento | 130 832    | 2007-2013 | 19.3%           |
| validacion    | 32 709     | 2007-2013 | 19.3%           |
| prueba        | 104 989    | 2014-2015 | 26.0%           |

*Fuente: elaboración propia (`riesgo/particiones.py`).*

La tasa de malos de la prueba (26.0 %) es bastante más alta que la de entrenamiento (19.3 %). La Tabla 4 y la Figura 1 muestran de dónde viene buena parte de esa diferencia: entre más reciente es el año, menos créditos tienen un desenlace conocido. De los originados en 2012, el 93.3 % ya había terminado en enero de 2016; de los de 2013, el 54.3 %; de los de 2014, el 31.5 %, y de los de 2015, apenas el 7.3 %. Un crédito a 36 meses otorgado en 2014 no alcanza a pagarse completo según el calendario antes de 2016. Los que ya tienen desenlace son los que terminaron antes de tiempo: o lo pagaron por adelantado o cayeron pronto en mora. Por eso, en los años recientes, los malos están sobrerrepresentados.

**Tabla 4.** Créditos por año de originación: cuántos tienen desenlace conocido, su tasa de malos y la proporción a 60 meses.

|   Año | Partición                  | Originados   | Con desenlace conocido   | % con desenlace   | Tasa de malos   | A 60 meses   |
|------:|:---------------------------|:-------------|:-------------------------|:------------------|:----------------|:-------------|
|  2007 | entrenamiento y validación | 603          | 603                      | 100.0%            | 26.2%           | 0.0%         |
|  2008 | entrenamiento y validación | 2 393        | 2 393                    | 100.0%            | 20.7%           | 0.0%         |
|  2009 | entrenamiento y validación | 5 281        | 5 281                    | 100.0%            | 13.7%           | 0.0%         |
|  2010 | entrenamiento y validación | 12 537       | 12 528                   | 99.9%             | 14.0%           | 26.9%        |
|  2011 | entrenamiento y validación | 21 721       | 19 729                   | 90.8%             | 16.5%           | 28.5%        |
|  2012 | entrenamiento y validación | 53 367       | 49 813                   | 93.3%             | 16.7%           | 12.8%        |
|  2013 | entrenamiento y validación | 134 755      | 73 194                   | 54.3%             | 22.9%           | 22.2%        |
|  2014 | prueba                     | 235 628      | 74 154                   | 31.5%             | 26.2%           | 27.5%        |
|  2015 | prueba                     | 421 094      | 30 835                   | 7.3%              | 25.5%           | 31.5%        |

*Fuente: elaboración propia con datos de Lending Club (ranadeep, 2021). "A 60 meses" se calcula sobre los créditos con desenlace conocido.*

![Figura 1](../resultados/descriptivo/tasa_por_anio.png)

**Figura 1.** Créditos con desenlace conocido (barras, eje izquierdo) y tasa de malos (línea, eje derecho) por año de originación. *Fuente: elaboración propia.*

A este efecto lo llamamos sesgo de madurez. Con estos datos no podemos separar cuánto del aumento de la tasa de malos es un deterioro real del riesgo y cuánto es selección por madurez. Lo tenemos en cuenta de tres maneras:

1. Las métricas de discriminación en prueba (AUC, Gini, KS) miden qué tan bien el modelo ordena a los clientes entre los créditos que terminaron rápido, que no es exactamente la población a la que se le prestaría.
2. Un modelo entrenado con una tasa de malos de 19.3 % va a quedarse corto, en promedio, en una prueba con 26.0 %. La curva de calibración de la sección 5 hay que leerla con eso en mente.
3. El sesgo también toca el entrenamiento: 2013 aporta el 45 % de los créditos de entrenamiento y validación, y de ese año solo tiene desenlace el 54.3 %.

Las alternativas, como cortar la prueba en 2013 o quedarse solo con créditos a 36 meses de años ya cerrados, dejan una prueba más limpia pero más vieja y más pequeña. Preferimos la partición que se parece al uso real del modelo y reportar el sesgo.

#### Referencias citadas en esta sección

ranadeep. (2021). *Credit risk analysis* [Conjunto de datos]. Kaggle. https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset
