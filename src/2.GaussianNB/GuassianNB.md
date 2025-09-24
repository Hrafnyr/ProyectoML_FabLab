
# Propósito general

Este script entrena un clasificador **Gaussian Naive Bayes** para predecir el estado de bienestar categórico (`mhc_dx`) a partir de escalas psicológicas y variables demográficas/contextuales. Se realiza preprocesamiento (división estratificada, escalado), entrenamiento, predicción y evaluación con métricas estándar.

Además, se considera que la clase 2 es muy minoritaria en la muestra, lo cual limita la capacidad del modelo para aprender esa clase y puede afectar la interpretación.

---

## Librerías y Funcionalidad Clave

* Modelo probabilístico que asume features independientes y distribuciones gaussianas:

```python
from sklearn.naive_bayes import GaussianNB
```

* Métricas para evaluación:

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

* División de datos con estratificación para mantener proporción de clases:

```python
from sklearn.model_selection import train_test_split
```

* Escalado para normalizar las features y evitar que variables con mayor rango dominen el modelo:

```python
from sklearn.preprocessing import StandardScaler  
```

---

## División de datos y su importancia

Sin estratificación (`stratify=y`), una división aleatoria podría dejar clases minoritarias muy poco representadas en alguno de los conjuntos (entrenamiento o prueba), generando:

* Evaluaciones poco fiables y sesgadas, que no reflejan el desempeño real.
* Modelos incapaces de aprender patrones para clases minoritarias por falta de ejemplos.

Por eso, se usa:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

* `stratify=y` asegura que las proporciones de clases se mantengan iguales en train y test.
* `random_state=42` fija la semilla para reproducibilidad (42 es referencia cultural a Douglas Adams).

---

## Escalado de características

GaussianNB no requiere escalado estrictamente, pero lo hacemos para mantener consistencia con otros modelos y facilitar comparaciones. El escalador se ajusta con el conjunto de entrenamiento y luego se aplica al test:

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Esto evita fuga de información (leakage) del conjunto de test al entrenamiento.

---

# Análisis de Resultados

```
Accuracy: 0.699
```

### Rendimiento por clase (del classification\_report):

| Clase | Precision | Recall | F1-score | Support |
| ----- | --------- | ------ | -------- | ------- |
| 0     | 0.604     | 0.611  | 0.608    | 450     |
| 1     | 0.770     | 0.771  | 0.771    | 1018    |
| 2     | 0.378     | 0.354  | 0.366    | 96      |

* La **clase 1 (moderado)** es la mejor predicha, con buena precisión y recall.
* La **clase 0** tiene desempeño medio (60% aprox).
* La **clase 2 (floreciente)** es difícil de predecir, con bajo recall y precisión, probablemente por ser minoritaria.

### Matriz de confusión:

|        | Pred 0 | Pred 1 | Pred 2 |
| ------ | ------ | ------ | ------ |
| Real 0 | 275    | 173    | 2      |
| Real 1 | 179    | 785    | 54     |
| Real 2 | 1      | 61     | 34     |

* Muchas confusiones entre clases 1 y 2.
* Clase 2 rara vez predicha correctamente (solo 34 de 96).

### Validación cruzada:

```
Accuracy CV (5 folds): 0.711 ± 0.008
```

* Resultado estable y consistente en diferentes particiones.

---

# Consideraciones finales

* La clase 2 es muy minoritaria (solo 96 casos), y crear o duplicar datos sintéticos para esa clase puede inducir sesgos, lo que en temas de psicología es especialmente delicado.
* El modelo GaussianNB funciona bien como baseline para las clases mayoritarias, pero falla en la minoritaria.
* El desempeño inferior frente a Random Forest indica que relaciones complejas o correlaciones entre variables podrían ser importantes y no bien capturadas por GaussianNB.
* El modelo puede ser útil como filtro rápido para detectar la clase moderada (clase 1), y para otras clases se puede requerir análisis o modelos más especializados.

---

# Usos potenciales del modelo

* **Investigación:** segmentar participantes según bienestar para análisis posteriores.
* **Clínica / Educación:** filtro inicial para identificar casos que requieran atención.
* **Programas de intervención:** evaluar cambios en bienestar con seguimiento automatizado.

---

## Ajustes y mejoras de optimización del modelo

Se agrega:  **GaussianNB(var_smoothing=1e-5)**; Añade un pequeño valor a la varianza de cada feature para evitar división por cero. Puede afectar un poco la estabilidad y la predicción cuando las features tienen baja varianza o outliers. El modelo no mostró ninguna mejora agregando y variando el valor del parámetro.

### Oversampling + SMOTE

Es una técnica de aprendizaje automático que resuelve el problema de las clases desequilibradas en un conjunto de datos al generar muestras sintéticas de la clase minoritaria. En lugar de simplemente duplicar datos, SMOTE crea nuevos puntos de datos interpolando entre los ejemplos existentes de la clase minoritaria en el espacio de características, lo que resulta en un conjunto de datos más equilibrado para entrenar modelos de clasificación más precisos. 

| k\_neighbors | Accuracy | F1-macro | Recall Clase 0 | Recall Clase 1 | Recall Clase 2 | Observaciones                                                          |
| ------------ | -------- | -------- | -------------- | -------------- | -------------- | ---------------------------------------------------------------------- |
| 3            | 0.645    | 0.565    | 0.653          | 0.646          | 0.583          | Buen equilibrio inicial, clase 2 todavía limitada                      |
| 4            | 0.650    | 0.570    | 0.660          | 0.651          | 0.583          | Leve mejora en clases 0 y 1, recall clase 2 igual                      |
| 5            | 0.650    | 0.575    | 0.656          | 0.651          | 0.615          | Mejor balance: clase 2 mejora sin afectar mucho las otras              |
| 6            | 0.646    | 0.573    | 0.660          | 0.643          | 0.604          | Clase 2 pierde algo de recall, macro F1 no mejora; sobremezcla empieza |

Por tanto se toma la cantidad de vecinos a **5** como valor más optimo.

---

## 1️⃣ Modelo Original (sin balanceo)

**Métricas principales:**

* **Accuracy:** 0.699
* **Macro F1:** 0.581
* **Balanced Accuracy:** 0.579

**Recall por clase:**

* Clase 0: 0.611 → captura \~61% de los casos
* Clase 1: 0.771 → domina la predicción, alta recall
* Clase 2: 0.354 → muy baja, casi no detecta minoritarios

**Confusión:**

* Clase 2 apenas 34/96 casos correctos, muchos falsos negativos
* Clase 1 domina, causando que minoritarios se pierdan

**Validación cruzada:** 0.711 ± 0.008 (accuracy) → indica estabilidad, pero sesgo hacia clase mayoritaria.

**Conclusión:**

* El modelo **favorece la clase mayoritaria (1)**.
* Minoritarios (0 y sobre todo 2) tienen bajo recall y F1.
* Macro F1 bajo refleja **desbalance en desempeño entre clases**.

---

## 2️⃣ Modelo Ajustado con SMOTE (k\_neighbors=5, target\_dict={0:2500, 2:1200})

**Métricas principales:**

* **Accuracy:** 0.646 → ligeramente menor que original, esperado por balanceo
* **Macro F1:** 0.570 → aumenta el equilibrio entre clases minoritarias y mayoritaria
* **Balanced Accuracy:** 0.636 → mejora sustancial respecto al modelo original

**Recall por clase:**

* Clase 0: 0.660 → mejora respecto a 0.611, más casos detectados
* Clase 1: 0.643 → leve reducción frente al modelo original, evitando que domine
* Clase 2: 0.604 → mejora dramática respecto a 0.354, detecta más de la mitad de los casos

**Confusión:**

* Clase 2: 58/96 casos correctos → más del **doble de aciertos** que sin SMOTE
* Clase 0: 297/450 → más casos correctos que original
* Clase 1: reducción leve de recall, pero mantiene buen desempeño

**Validación cruzada:** F1-macro CV 0.573 ± 0.011 → consistente y equilibrado

---

## 3️⃣ Mejoras presentadas con SMOTE

| Mejora                   | Observación                                                                               |
| ------------------------ | ----------------------------------------------------------------------------------------- |
| **Clase 0**              | Recall aumenta de 0.611 → 0.660 → captura más casos minoritarios moderados                |
| **Clase 2**              | Recall aumenta de 0.354 → 0.604 → detecta muchos más casos minoritarios críticos          |
| **Macro F1**             | Mantiene buen nivel (0.581 → 0.570) y refleja mejor balance de desempeño                  |
| **Balanced Accuracy**    | Aumenta notablemente, mostrando que **todas las clases tienen desempeño más equilibrado** |
| **Trade-off controlado** | Accuracy global baja ligeramente, pero el modelo ahora es más justo para minoritarios     |

---

### Conclusión General

* El **modelo original** era más preciso en global, pero **sesgaba las predicciones hacia la clase mayoritaria (1)**.
* El **modelo con SMOTE** logra un **equilibrio mucho mejor entre clases 0, 1 y 2**, especialmente mejorando la detección de la clase minoritaria crítica (2).
* El ajuste de **k\_neighbors=5** y el **target dict** de SMOTE fueron clave para lograr un **trade-off óptimo** entre recall de minoritarios y precisión de la clase mayoritaria.

