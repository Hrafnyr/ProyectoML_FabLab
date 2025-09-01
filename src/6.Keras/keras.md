# Clasificación de Bienestar Mental con Redes Neuronales (Keras)

## 1. Propósito

El objetivo de este modelo es **predecir el estado de bienestar mental** (`mhc_dx`) de los individuos a partir de una combinación de escalas psicológicas y variables demográficas.  
El dataset contiene 7,500 registros con 11 features, y la variable objetivo tiene tres categorías:

- 0 → Desanimado
- 1 → Moderado
- 2 → Florecido

Se utilizó un enfoque de **red neuronal multicapa (MLP) con Keras**, entrenada para clasificar correctamente a los individuos en una de las tres categorías.

---

## 2. Configuración del Modelo

### 2.1 Preprocesamiento

- **Train/Test split:** 80/20, estratificado según `mhc_dx`.
- **Escalado:** `StandardScaler` aplicado a todas las features.
- **Batch size:** 32
- **Epochs:** 150
- **Validación interna:** 20% del train para monitorizar el overfitting.

### 2.2 Arquitectura de la Red

| Capa | Unidades | Activación |
|------|----------|------------|
| Input | 12 (features) | ReLU |
| Hidden 1 | 12 | ReLU |
| Hidden 2 | 8 | ReLU |
| Hidden 3 | 6 | ReLU |
| Output | 3 | Softmax |

### 2.3 Compilación

- **Función de pérdida:** `sparse_categorical_crossentropy` (porque la variable objetivo es numérica: 0, 1, 2)
- **Optimizador:** Adam con `learning_rate=0.001`
- **Métrica de evaluación:** Accuracy

---

## 3. Resultados

### 3.1 Precisión global

- **Accuracy en test:** 0.733 (~73%)

### 3.2 Reporte de clasificación

| Clase | Precision | Recall | F1-score | Support |
|-------|----------|--------|----------|---------|
| 0 (Desanimado) | 0.65 | 0.55 | 0.60 | 450 |
| 1 (Moderado) | 0.76 | 0.87 | 0.81 | 1018 |
| 2 (Florecido) | 0.71 | 0.16 | 0.26 | 96 |

**Promedios:**

- Macro avg: precision 0.71, recall 0.53, f1 0.55  
- Weighted avg: precision 0.73, recall 0.73, f1 0.71

### 3.3 Matriz de confusión

[[249 201 0]

[130 882 6]

[ 2 79 15]]


---

## 4. Análisis e Interpretación

1. **Clase mayoritaria (Moderado):**  
   - Excelente desempeño, recall de 0.87 → la mayoría de individuos moderados son correctamente identificados.

2. **Clase Desanimado:**  
   - Recall de 0.55 → la mitad de los casos se identifican correctamente, algunos se confunden con Moderado.  
   - Esto indica que la red tiene más dificultad para diferenciar Desanimado de Moderado.

3. **Clase Florecido:**  
   - Recall muy bajo (0.16) a pesar de buena precision (0.71).  
   - El modelo tiene problemas para identificar correctamente individuos Florecidos, probablemente por el **desbalance del dataset** (solo 96 casos de 7,500).

4. **Conclusión general:**  
   - El modelo alcanza un **accuracy global aceptable (~73%)**, pero la capacidad de detectar casos minoritarios (Florecido) es limitada.  
   - Se podría mejorar el rendimiento en clases minoritarias usando **class weights**, técnicas de oversampling (SMOTE) o recolectando más datos para equilibrar las clases.

---

## 5. Notas adicionales

- El tamaño del dataset (~7,500 registros) es adecuado para redes neuronales simples (MLP), pero insuficiente para modelos profundos complejos.
