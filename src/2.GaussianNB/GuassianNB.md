
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
