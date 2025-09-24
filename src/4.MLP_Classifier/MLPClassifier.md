# Clasificación de Bienestar Psicológico con MLPClassifier

## Propósito general

Este script entrena un clasificador MLP (Multi-layer Perceptron) para predecir el estado de bienestar categórico (`mhc_dx`) a partir de escalas psicológicas y variables demográficas/contextuales. Se realiza la división de datos en entrenamiento y prueba, preprocesamiento (escalado), entrenamiento, predicción y evaluación con métricas estándar.

---

## Análisis del código

1. **Carga de datos:** Se importa el dataset desde un archivo CSV.
2. **Selección de características y target:** Se eligen 10 variables predictoras relevantes (escalas psicológicas y datos demográficos) y la variable objetivo `mhc_dx`.
3. **División del dataset:** Se separan los datos en conjuntos de entrenamiento (80%) y prueba (20%), con estratificación para mantener la distribución original de clases.
4. **Preprocesamiento:** Se normalizan las características usando `StandardScaler` para que tengan media 0 y desviación estándar 1, lo que mejora el desempeño del MLP.
5. **Entrenamiento:** Se entrena un modelo `MLPClassifier` con dos capas ocultas de tamaños 30 y 20, activación logística, regularización (`alpha=0.01`), máximo 3000 iteraciones, y semilla fija para reproducibilidad.
6. **Evaluación:** Se predicen las etiquetas en el conjunto de prueba y se calculan métricas de desempeño.
7. **Persistencia:** El modelo entrenado se guarda para uso posterior.

---

## Análisis de resultados

### Precisión global

```

Precisión del modelo: 0.7404

```

- El modelo acertó aproximadamente el 74% de las predicciones en datos no vistos (test).

### Reporte detallado por clase

| Clase | Precision | Recall | F1-score | Soporte |
|-------|-----------|--------|----------|---------|
| 0     | 0.66      | 0.58   | 0.62     | 450     |
| 1     | 0.77      | 0.87   | 0.81     | 1018    |
| 2     | 0.77      | 0.18   | 0.29     | 96      |

- **Clase 0 (Languishing):** El modelo detecta bien la mayoría (recall 0.58), aunque con algunas falsas predicciones (precision 0.66).
- **Clase 1 (Moderado):** Mejor desempeño, con alta precisión y recall, mostrando buena capacidad para identificar correctamente esta clase mayoritaria.
- **Clase 2 (Floreciente):** Precision alta (0.77) pero recall muy bajo (0.18), indica que el modelo es conservador en predecir esta clase y a menudo no detecta muchos casos reales de clase 2.

### Matriz de confusión

```

[[259 191   0]
[131 882   5]
[  1  78  17]]

```

- Hay confusión considerable entre la clase 0 y 1, con falsos negativos y positivos.
- La clase 2 se detecta poco (sólo 17 de 96 casos correctos), muchos son clasificados erróneamente como clase 1.
- Esto refleja la dificultad de distinguir la clase minoritaria con los datos y características actuales.

---

## Conclusiones y recomendaciones para implementación

- El modelo MLP logra un desempeño sólido en la clasificación general del bienestar psicológico, especialmente para la clase moderada (clase 1).
- La baja detección (recall) de la clase minoritaria (floreciente) sugiere que:
  - Podría beneficiarse de técnicas específicas para balancear clases (sobremuestreo, pesos de clase).
  - Más datos o mejores features ayudarían a mejorar la discriminación.
- La regularización y arquitectura actual ayudan a evitar overfitting y generan un modelo estable.
- Para aplicaciones prácticas, este modelo puede usarse como filtro inicial para segmentar automáticamente la mayoría de casos en clase moderada.
- Para la clase minoritaria, sería prudente usar un segundo análisis o evaluación complementaria debido al bajo recall.
- Se recomienda validar y ajustar continuamente con datos nuevos y explorar hiperparámetros adicionales o técnicas de ensamble para mejorar el desempeño global y por clase.

---

## Ajustes y mejoras de optimización del modelo

Perfecto 👍 Sí, tengo todo lo que probaste guardado en esta conversación. Podemos organizarlo en una **tabla resumen** para documentar claramente cada escenario de prueba del modelo MLP con SMOTE, incluyendo los parámetros clave, resultados de accuracy, F1-score y observaciones.

Aquí te propongo la estructura en **Markdown**:

---

### 📊 Resumen Experimentos MLPClassifier + SMOTE

| Escenario            | Capas (hidden\_layer\_sizes) | Iteraciones | Activación | Solver | Alpha | SMOTE sampling\_strategy | Accuracy | F1-macro | Recall clase 0 | Recall clase 1 | Recall clase 2 | Observaciones                                                         |
| -------------------- | ---------------------------- | ----------- | ---------- | ------ | ----- | ------------------------ | -------- | -------- | -------------- | -------------- | -------------- | --------------------------------------------------------------------- |
| Original (sin SMOTE) | (30,20)                      | 3000        | logistic   | adam   | 0.01  | Ninguno                  | 0.7404   | 0.57     | 0.58           | 0.87           | 0.18           | Muy buena precisión global pero clase minoritaria (2) muy baja.       |
| SMOTE + logistic     | (30,20)                      | 3000        | logistic   | adam   | 0.01  | {0:2500, 2:1200}         | 0.6764   | 0.54     | 0.59           | 0.75           | 0.30           | Mejora recall clase 2, pierde algo de accuracy general.               |
| logistic             | (50,30)                      | 3000        | logistic   | adam   | 0.01  | {0:2500, 2:1200}         | 0.6918   | 0.57     | 0.57           | 0.77           | 0.41           | Mejor balance general, especialmente clase 2.                         |
| tanh                 | (50,30)                      | 3000        | tanh       | adam   | 0.01  | {0:2500, 2:1600}         | 0.6329   | 0.51     | 0.54           | 0.70           | 0.32           | Caída en performance, no recomendado.                                 |
| relu                 | (50,36)                      | 3000        | relu       | sgd    | 0.01  | {0:2500, 2:1600}         | 0.6329   | 0.51     | 0.54           | 0.70           | 0.32           | Similar a tanh, peor recall clase 2.                                  |
| logistic             | (60,30)                      | 3000        | logistic   | sgd    | 0.01  | {0:2500, 2:1200}         | 0.7078   | 0.61     | 0.64           | 0.76           | 0.44           | Mejor performance global, recall clase 2 sube.                        |
| logistic             | (60,40)                      | 3000        | logistic   | adam   | 0.01  | {0:2500, 2:1200}         | 0.6847   | 0.56     | 0.55           | 0.77           | 0.41           | No supera el anterior, menor recall clase 0.                          |
| logistic             | (80,40)                      | 3000        | logistic   | sgd    | 0.01  | {0:2500, 2:1200}         | 0.7078   | 0.61     | 0.64           | 0.76           | 0.46           | Rendimiento muy estable, buena mejora clase minoritaria.              |
| logistic             | (80,40)                      | 2000        | logistic   | sgd    | 0.01  | {0:2500, 2:1200}         | 0.7078   | 0.61     | 0.64           | 0.76           | 0.46           | Igual performance, menos iteraciones = mayor eficiencia.              |
| logistic             | (100,60)                     | 1000        | logistic   | sgd    | 0.01  | {0:2500, 2:1200}         | 0.7078   | 0.61     | 0.64           | 0.76           | 0.46           | Igual performance que anterior, aumenta complejidad innecesariamente. |

Tabla de pruebas realizadas para encontrar solución más óptima.

### Conclusiones generales

* **SMOTE** ayuda especialmente a mejorar el recall de la clase minoritaria (clase 2), aunque puede reducir accuracy general.
* **Activación logistic + solver sgd + capas (80,40)** ofrece un balance sólido entre recall, precisión y accuracy.
* **Reducir iteraciones** no impacta negativamente en performance, permitiendo ahorrar tiempo de entrenamiento.
* **Capas muy grandes** o iteraciones excesivas no generan mejoras significativas y pueden causar overfitting.
* Este escenario final **(80,40, logistic, sgd, 2000 iter)** es óptimo para balancear precisión global y desempeño en todas las clases.

---

## Comparativa: Modelo Original vs Modelo Óptimo con SMOTE

| Métrica               | Modelo Original | Modelo Óptimo (SMOTE + MLP) |
| --------------------- | --------------- | --------------------------- |
| **Accuracy**          | 0.7404          | 0.7084                      |
| **Macro F1-score**    | 0.57            | 0.62                        |
| **Recall Clase 0**    | 0.58            | 0.64                        |
| **Recall Clase 1**    | 0.87            | 0.76                        |
| **Recall Clase 2**    | 0.18            | 0.47                        |
| **Precision Clase 0** | 0.66            | 0.61                        |
| **Precision Clase 1** | 0.77            | 0.79                        |
| **Precision Clase 2** | 0.77            | 0.44                        |

---

### Matrices de Confusión

**Modelo Original:**

```
[[259 191   0]
 [131 882   5]
 [  1  78  17]]
```

**Modelo Óptimo (SMOTE + MLP):**

```
[[288 162   0]
 [185 775  58]
 [  1  50  45]]
```

---

### Análisis de Resultados

1. **Accuracy**

   * El modelo original obtiene mayor accuracy global (0.74 vs 0.71), pero este valor es engañoso porque no refleja bien el desempeño en la clase minoritaria.

2. **Recall por clase**

   * **Clase 2 (minoritaria)**: mejora drástica con SMOTE (de 0.18 → 0.47), lo cual es clave si queremos clasificar bien casos poco frecuentes.
   * **Clase 0** también mejora ligeramente (0.58 → 0.64).
   * **Clase 1** disminuye un poco (0.87 → 0.76), pero sigue alto, mostrando buen balance general.

3. **F1-score macro**

   * Mejora de **0.57 → 0.62**, indicando que el modelo óptimo balancea mejor precisión y recall entre clases.

4. **Precisión por clase**

   * Clase 2 cae en precisión (0.77 → 0.44), pero ganamos mucho recall, lo que es más importante para detectar correctamente casos minoritarios.

5. **Matrices de Confusión**

   * El modelo original clasifica muy bien la clase mayoritaria (clase 1) pero casi ignora la clase minoritaria (clase 2: solo 17 aciertos sobre 96).
   * El modelo óptimo mejora considerablemente la detección de la clase 2 (45 aciertos sobre 96), sacrificando algo de accuracy general.

## Hallazgos relevantes

* **SMOTE + ajuste de arquitectura MLP** logra un modelo más equilibrado para todas las clases, mejorando especialmente la detección de la clase minoritaria, que es crítico para problemas desbalanceados.
* Aunque la **accuracy total disminuye ligeramente**, la ganancia en recall y F1-score macro justifica la mejora, ya que el modelo no prioriza únicamente la clase mayoritaria.
* El nuevo modelo es una opción óptima para implementación, especialmente considerando el **minimizar falsos negativos en la clase minoritaria**.
