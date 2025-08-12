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
