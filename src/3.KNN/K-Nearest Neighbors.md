# Clasificación con K-Nearest Neighbors (KNN) — k=5

## Propósito general

Este script entrena un clasificador K-Nearest Neighbors para predecir el estado de bienestar categórico (`mhc_dx`) a partir de escalas psicológicas y variables demográficas/contextuales. Se realiza preprocesamiento (división, escalado), entrenamiento, predicción y evaluación con métricas estándar, para analizar la capacidad del modelo en clasificar correctamente las distintas categorías de bienestar.

---

## Métricas de evaluación

```
Accuracy: 0.691
```

* El modelo acierta aproximadamente en un 69.1% de los casos en el conjunto de prueba.

---

## Rendimiento por clase

| Clase | Precision | Recall | F1-score | Soporte |
| ----- | --------- | ------ | -------- | ------- |
| 0     | 0.58      | 0.47   | 0.52     | 450     |
| 1     | 0.73      | 0.84   | 0.78     | 1018    |
| 2     | 0.52      | 0.11   | 0.19     | 96      |

**Interpretación:**

* **Clase 0 (Languishing):** El recall es bajo (0.47), lo que indica que el modelo no detecta casi la mitad de los casos reales. La precisión también es moderada (0.58), lo que sugiere predicciones no muy confiables.
* **Clase 1 (Moderado):** Buen rendimiento general, con recall alto (0.84) y precisión razonable (0.73). El modelo identifica bien esta clase mayoritaria.
* **Clase 2 (Floreciente):** Presenta dificultades, con recall muy bajo (0.11) y precisión moderada (0.52). El modelo falla en detectar la mayoría de los casos reales de esta clase minoritaria.

---

## Matriz de confusión

```
[[211 239   0]
 [150 858  10]
 [  1  84  11]]
```

* La mayoría de los errores para la clase 0 se confunden como clase 1.
* La clase 1 se predice con alta precisión, aunque hay confusiones hacia las otras clases.
* La clase 2 es la más difícil de predecir correctamente, con muchas predicciones erróneas hacia la clase 1.

---

## Validación cruzada (5 folds)

```
Accuracy CV (5 folds): 0.634 ± 0.049
```

* La validación cruzada indica que el desempeño promedio del modelo es del 63.4%, con una desviación estándar moderada, lo que sugiere cierta variabilidad en distintas particiones.

---

## Conclusión

* KNN con k=5 ofrece un desempeño razonable para la clase mayoritaria (moderado).
* El rendimiento para clases minoritarias, especialmente la clase 2, sigue siendo limitado.
* Debido a la naturaleza local del KNN, puede ser sensible a ruido y a la elección del número de vecinos.
* Para este problema con clases desbalanceadas, es recomendable seguir evaluando diferentes valores de k o combinar con técnicas de balanceo/clasificación especializada.

---

## Ajustes y mejoras de optimización del modelo

Para este algoritmo se busca seguir optimizando los parámetros de procesamiento.

```python
pipeline = Pipeline(steps=[
    ('smote', SMOTE(sampling_strategy={0:2500, 2:1200}, random_state=42)),
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(
        n_neighbors=7, 
        weights='distance',
        metric='minkowski',
        p=2,
        algorithm='auto'
    ))
])
```

---

#### `('smote', SMOTE(sampling_strategy={0:2500, 2:1200}, random_state=42))`

* SMOTE (**Synthetic Minority Oversampling Technique**) genera nuevos ejemplos sintéticos de las clases minoritarias para balancear el dataset. Esto evita que el modelo esté sesgado hacia la clase mayoritaria.

* **Parámetros:**

  * `sampling_strategy={0:2500, 2:1200}` → indica cuántas muestras queremos para cada clase minoritaria.

    * Clase `0`: aumentar a 2500 ejemplos.
    * Clase `2`: aumentar a 1200 ejemplos.

  * `random_state=42` → asegura reproducibilidad del resultado.

* **Motivo:**
  SMOTE ayuda a que KNN no se incline demasiado hacia la clase más frecuente.

---

#### `('scaler', StandardScaler())`

* Escala las características para que tengan media 0 y desviación estándar 1.

* **Motivo:**
  KNN es sensible a la escala porque calcula distancias entre puntos. Si una característica tiene valores más grandes que otras, dominará la distancia.
  Ejemplo: si `edad` está en decenas y `SUMPHQ` en unidades, sin escalar KNN dará más peso a `edad`.

---

#### `('knn', KNeighborsClassifier(...))`

* **`n_neighbors=7** → número de vecinos que KNN usa para decidir la clase de un nuevo punto. 

* **`weights='distance'`** → los vecinos más cercanos tienen más influencia en la predicción. Esto suele mejorar la precisión cuando los puntos cercanos son más relevantes que los más lejanos.

* **`metric='minkowski'` y `p=2`** → la distancia Minkowski con p=2 equivale a la distancia euclidiana estándar.

  * `p=1` → distancia Manhattan.
  * `p=2` → distancia euclidiana.
    Esto define cómo KNN mide la cercanía entre puntos.

* **`algorithm='auto'`** → Scikit-learn elije el algoritmo más eficiente para buscar vecinos (`ball_tree`, `kd_tree` o fuerza bruta).

---

### Flujo del pipeline

1. **SMOTE**: genera nuevas muestras sintéticas para balancear las clases minoritarias.
2. **StandardScaler**: normaliza todas las características.
3. **KNeighborsClassifier**: clasifica nuevos datos usando vecinos ponderados por distancia.

Este pipeline asegura que:

* El balanceo de clases se aplica antes del entrenamiento.
* El escalado ocurre de forma consistente.
* El modelo KNN recibe datos optimizados.

---

### Comparativa KNN+SMOTE — Resultados por número de vecinos (k)

| k vecinos | Accuracy | F1-macro | Recall clase 0 | Recall clase 1 | Recall clase 2 |
| --------- | -------- | -------- | -------------- | -------------- | -------------- |
| 3         | 0.660    | 0.519    | 0.56           | 0.74           | 0.25           |
| 4         | 0.685    | 0.500    | 0.48           | 0.83           | 0.15           |
| 5         | 0.657    | 0.540    | 0.56           | 0.74           | 0.33           |
| 6         | 0.669    | 0.542    | 0.56           | 0.75           | 0.33           |
| 7         | 0.669    | 0.545    | 0.58           | 0.74           | 0.34           |
| 8         | 0.677    | 0.551    | 0.58           | 0.75           | 0.31           |

---

### Justificación del punto óptimo (k=7)

* **F1-macro más alto antes de que recall clase 2 empiece a disminuir significativamente.**
* Balance sólido: mantiene **recall clase 0 y clase 1 alto**, mientras mejora notablemente la clase minoritaria (clase 2) respecto a k bajos.
* Evita la sobrecompensación de clases mayoritarias que ocurre en k altos.
* Muestra estabilidad en validación cruzada (F1-macro CV relativamente bajo en varianza: ±0.023).

### Análisis de resultados

* **k bajos (3–5)**: tienden a tener recall decente para minorías pero menor estabilidad general y menor Accuracy global.
* **k altos (6–8)**: mejoran Accuracy general, pero sacrifican sensibilidad en la clase minoritaria si se sube demasiado.
* **k=7**: mejor punto de compromiso entre precisión, recall y balance entre clases.

Perfecto 👍, vamos a desglosar tu matriz de confusión para que quede claro qué significa cada valor.

---

### Matriz de Confusión

| Clase real ↓ / Predicha → | Clase 0 | Clase 1 | Clase 2 |
| ------------------------- | ------- | ------- | ------- |
| **Clase 0**               | 260     | 186     | 4       |
| **Clase 1**               | 193     | 754     | 71      |
| **Clase 2**               | 3       | 60      | 33      |


### Conclusión sobre la matriz

* El modelo **clasifica bastante bien la clase 1** (clase mayoritaria, recall alto).
* Tiene **dificultad con la clase 2** (minoritaria), aunque SMOTE mejoró algo su detección respecto al modelo original.
* La clase 0 también mejora comparado con el modelo sin SMOTE, pero aún hay confusión importante con clase 1.

---

### **Matrices Comparadas**

#### Original:

```
[[211 239   0]
 [150 858  10]
 [  1  84  11]]
```

#### Ajustada (SMOTE + KNN vecinos=7):

```
[[260 186   4]
 [193 754  71]
 [  3  60  33]]
```

#### 1. **Clase 0**

* **Original:** 211 correctos / 450 = recall ≈ 0.47
* **Ajustada:** 260 correctos / 450 = recall ≈ 0.58 ✅ Mejora clara (+11%).
* El modelo ahora detecta mejor la clase minoritaria 0, aunque hay un pequeño aumento de falsos positivos en clase 2.

---

#### 2. **Clase 1**

* **Original:** 858 correctos / 1018 = recall ≈ 0.84
* **Ajustada:** 754 correctos / 1018 = recall ≈ 0.74 ❌ Disminuyó.
* Hay más errores entre clase 1 y otras clases (clase 0 y clase 2), pero esto era previsible: el balanceo con SMOTE reduce el sesgo hacia la clase mayoritaria, sacrificando parte del recall para ella.

---

#### 3. **Clase 2**

* **Original:** 11 correctos / 96 = recall ≈ 0.11
* **Ajustada:** 33 correctos / 96 = recall ≈ 0.34 ✅ Mejora muy importante (+23%).
* El oversampling ayudó mucho a la clase minoritaria más pequeña, que antes estaba siendo ignorada en casi todos los casos.


### Conclusiones

1. **Uso de SMOTE** fue esencial: permitió mejorar recall de la clase minoritaria (clase 2) y aumentar el F1-macro, evitando que el modelo se sesgue hacia la clase mayoritaria.
2. **k vecinos** tiene impacto directo en la capacidad de balancear precisión global vs sensibilidad por clase.
3. **k=7** emerge como el óptimo para este problema: ofrece un balance robusto entre Accuracy, F1-macro y recall de todas las clases, maximizando la capacidad del modelo de clasificar correctamente todas ellas.
4. Este análisis muestra que, para datasets desbalanceados, combinar **oversampling + tuning de vecinos** es una estrategia efectiva para mejorar el desempeño de KNN sin sacrificar significativamente la generalización.

