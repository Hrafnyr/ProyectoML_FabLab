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

El tamaño del dataset (~7,500 registros) es adecuado para redes neuronales simples (MLP), pero insuficiente para modelos profundos complejos. Además se requiere instalar el paquete que es bastante pesado, puede probarse implementaciones en Notebooks.

**Funciones de activación.**

 ReLU (Rectified Linear Unit), se utilizó porque es la más rápida de implementar, evita en gran medida el problema del gradiente desvanecido (el gradiente se irá desvaneciendo a valores muy pequeños) que pasa con sigmoid o tanh y permite que la red aprenda relaciones no lineales complejas. 

Se usa en capas ocultas porque es el estándar actual en redes profundas. Acelera la convergencia en entrenamiento. Funciona bien en datos continuos y escalados como es el caso. 

**Softmax** en la última capa, convierte las salidas en probabilidades entre 0 y 1. Dado que el problema tiene 3 clases mutuamente excluyentes (Languishing, Moderado, Flourishing), softmax obliga al modelo a clasificar por una sola clase en este caso, la más probable.

Esto permite interpretar fácilmente las predicciones y usar métricas como categorical_crossentropy para determinar la clase predicha.

**Explicación de las capas y parámetros**

1. **Primera capa densa**

   ```python
   model.add(Dense(12, input_dim=X_train_scaled.shape[1], activation='relu'))
   ```

   * **Dense(12)** → es una **capa totalmente conectada** con 12 neuronas.
   * **input\_dim=X\_train\_scaled.shape\[1]** → el número de *features de entrada* (11 variables seleccionadas). Esto le dice a Keras cuántos valores recibe cada muestra.
   * **activation='relu'** → la función de activación aplicada a cada neurona.
   * **Propósito**: recibe los 11 features originales y los transforma en una primera representación de 12 dimensiones que la red puede procesar mejor.

2. **Segunda capa densa**

   ```python
   model.add(Dense(8, activation='relu'))
   ```

   * **Dense(8)** → 8 neuronas conectadas a las 12 de la capa anterior.
   * **activation='relu'** → mantiene no linealidad.
   * **Propósito**: reducir la dimensionalidad interna (de 12 → 8) mientras aprende patrones más abstractos.
   * Esta capa ya empieza a **combinar relaciones** entre las variables de entrada.

3. **Tercera capa densa**

   ```python
   model.add(Dense(6, activation='relu'))
   ```

   * **Dense(6)** → 6 neuronas, conectadas a las 8 anteriores.
   * **activation='relu'** → igual que antes, mantiene capacidad de aprender relaciones no lineales.
   * **Propósito**: seguir compactando la representación interna (de 8 → 6) para evitar sobreajuste y obligar al modelo a aprender **representaciones más eficientes** de los datos.


4. **Capa de salida**

   ```python
   model.add(Dense(3, activation='softmax'))
   ```

   * **Dense(3)** → 3 neuronas, una por cada clase de la variable objetivo `mhc_dx` (0 = Languishing, 1 = Moderado, 2 = Flourishing).
   * **activation='softmax'** → transforma las salidas en **probabilidades normalizadas** que suman 1.
   * **Propósito**: dar la predicción final del modelo como una probabilidad para cada clase → se elige la de mayor valor.

   **Diseño de la arquitectura** estructura tipo **embudo** (12 → 8 → 6 → 3). Esto ayuda a:

   * Empezar con más neuronas para capturar interacciones.
   * Ir reduciendo poco a poco para forzar al modelo a **condensar la información**.
   * Terminar con 3 neuronas que representan directamente las 3 clases.

**Compilación del modelo**

   ```python
   model.compile(
      loss='sparse_categorical_crossentropy',  
      optimizer=Adam(learning_rate=0.001),     
      metrics=['accuracy']
   )
   ```

### 1. **loss = 'sparse\_categorical\_crossentropy'**

El loss es una medida de qué tan mal o bien está prediciendo el modelo.

Durante el entrenamiento, la red intenta minimizar el loss, es decir, hacer que las predicciones se acerquen lo más posible a los valores reales.

En clasificación, el loss indica qué tan lejos están las probabilidades predichas de las clases correctas. Se usa en problemas de **clasificación multiclase** (más de 2 categorías).

“Sparse” significa que el **target `y` está en formato de enteros (0, 1, 2)**, no en **one-hot encoding**.

Calcula qué tan lejos están las predicciones del modelo de las clases reales.

Si los labels estuvieran en one-hot, se usa `categorical_crossentropy`. Ejemplo para 3 clases (0, 1, 2):

      0 → [1, 0, 0]

      1 → [0, 1, 0]

      2 → [0, 0, 1]


**optimizer = Adam(learning\_rate=0.001)**

* **Adam** = Adaptive Moment Estimation, combina ventajas de **SGD** y **RMSProp**.
* Se adapta automáticamente al gradiente y tasa de aprendizaje.
* `learning_rate=0.001` es el valor por defecto recomendado, pero se puede experimentar con valores mayores (0.01) o menores (0.0001) según la convergencia.

**metrics = \['accuracy']**

* Pide que se calcule la **precisión** (accuracy) durante el entrenamiento y validación.
* Es la métrica más simple para clasificación: porcentaje de predicciones correctas.

**Entrenamiento del modelo**

```python
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=150,
    batch_size=32,
    verbose=1
)
```

**validation\_split=0.2**

* Usa un **20% de los datos de entrenamiento** como conjunto de validación.
* Permite monitorear el desempeño del modelo en datos no vistos durante el entrenamiento.

**epochs=150**

* Una **época** = una pasada completa de todos los datos de entrenamiento a través de la red.
* Con 150 épocas, el modelo ve 150 veces todo el dataset.
* Más épocas = más oportunidades de aprender, pero también riesgo de **overfitting**.
* Normalmente se combina con **early stopping** para detenerse si la validación ya no mejora.

**batch\_size=32**

* Los datos no se procesan todos de una vez, sino en **lotes (batches)**.
* Cada batch contiene **32 muestras** que se pasan por la red antes de actualizar los pesos.
* Valores comunes: 16, 32, 64.

  * **Más pequeño (16)** → entrenamiento más preciso, pero más lento.
  * **Más grande (64 o 128)** → entrenamiento más rápido, pero menos preciso. Se dejó un valor intermedio para iniciar.

**verbose=1**

* Nivel de detalle en pantalla durante el entrenamiento.
* `1` = muestra una barra de progreso por época.
* `0` = no muestra nada.
* `2` = una línea por época, sin barra.

---

## **Optimizadores en Keras/TensorFlow**

El optimizador es el algoritmo que **actualiza los pesos de la red** durante el entrenamiento para minimizar el loss.

### 1. **Adam (Adaptive Moment Estimation)**

* Combina **momentum** y **adaptación de la tasa de aprendizaje** por parámetro.
* Muy usado por su **rapidez y estabilidad**.
* `learning_rate` controla cuánto cambian los pesos en cada paso.

### 2. **SGD (Stochastic Gradient Descent)**

* Gradiente descendente clásico con **actualizaciones por lotes (batches)**.
* Puede usar **momentum**, que ayuda a **acelerar la convergencia y evitar oscilaciones**:

  ```python
  optimizer = SGD(learning_rate=0.01, momentum=0.9)
  ```
* Momentum = combinación de la dirección actual y la anterior del gradiente.

### 3. **RMSprop**

* Ajusta automáticamente la **tasa de aprendizaje por cada parámetro**.
* Bueno para problemas con **gradientes que cambian mucho** o datos secuenciales.

### 4. **Adagrad**

* Ajusta el learning rate según la frecuencia de actualización de cada parámetro.
* Parámetros que cambian poco reciben un **learning rate más grande**, los que cambian mucho, más pequeño.
* Útil para datos dispersos.

### 5. **Adadelta**

* Variante de Adagrad que **limita la disminución del learning rate**.
* Evita que se vuelva demasiado pequeño con el tiempo.

---

### **Momentum**

* Es una **propiedad de algunos optimizadores (SGD, RMSprop, Adam)**.
* Ayuda a que los pesos se muevan más suavemente hacia el mínimo, **acelerando la convergencia** y reduciendo oscilaciones.
* En Adam, se aplica de forma automática a través de sus momentos del gradiente.

