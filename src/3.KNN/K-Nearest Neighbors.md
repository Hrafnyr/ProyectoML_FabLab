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
