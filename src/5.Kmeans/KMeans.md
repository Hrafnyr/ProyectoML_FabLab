# Análisis de Clustering con KMeans sobre Bienestar Mental (`mhc_dx`)

## Propósito

El objetivo de este código es aplicar un algoritmo de clustering no supervisado (KMeans) para agrupar individuos según múltiples características clínicas, demográficas y sociales relacionadas con su bienestar mental (`mhc_dx`). El fin es identificar grupos naturales en los datos que permitan descubrir patrones latentes y relacionarlos con las categorías de bienestar mental.

---

## Aspectos Clave del Código

- **Selección de variables (`features`):**  
  Se utilizan variables clínicas como `SUMPHQ`, `SumaGAD`, `mhc_total`, y otras escalas relacionadas con el bienestar emocional, junto con variables demográficas y sociales como edad, sexo, estado civil, consumo de sustancias, entre otras (un total de 18 variables).

- **Preprocesamiento:**  
  Los datos son escalados con `StandardScaler` para que todas las variables tengan media cero y varianza uno, facilitando que KMeans use distancias equitativamente.

- **Entrenamiento del modelo KMeans:**  
  Se define la cantidad de clusters (`n_clusters=3`) y se ajusta el modelo a los datos escalados.

- **Predicción y análisis:**  
  Se asigna a cada individuo un cluster y se analiza la distribución relativa (`proporciones`) y absoluta (`conteos`) de la variable categórica `mhc_dx` en cada cluster.

- **Visualización:**  
  Se genera un gráfico de barras apiladas mostrando la proporción de cada categoría `mhc_dx` dentro de cada cluster, para facilitar la interpretación.

- **Guardado de modelos:**  
  Tanto el modelo KMeans como el scaler se guardan para su uso futuro en inferencia.

---

## Resultados del Análisis



### Distribución de `mhc_dx` por cluster (proporciones)
| Cluster | Desanimado (0) | Moderado (1) | Florecido (2) |
|---------|----------------|--------------|---------------|
| 0       | 0.114          | 0.760        | 0.125         |
| 1       | 0.192          | 0.759        | 0.049         |
| 2       | 0.582          | 0.410        | 0.008         |

### Conteo absoluto
| Cluster | Desanimado (0) | Moderado (1) | Florecido (2) | Total |
|---------|----------------|--------------|---------------|-------|
| 0       | 293            | 1952         | 322           | 2567  |
| 1       | 543            | 2143         | 139           | 2825  |
| 2       | 1412           | 996          | 19            | 2427  |

---

## Análisis de los Clusters

- **Cluster 0 – Moderados/Florecidos**
  - Mayoría moderados, con proporción relevante de florecidos.  
  - Representa un grupo con **funcionamiento positivo** y cierta resiliencia.

- **Cluster 1 – Moderados/Intermedios**
  - Predominan moderados, pero con menos florecidos y más desanimados que el cluster 0.  
  - Puede considerarse un grupo de **riesgo medio**, con vulnerabilidad latente.

- **Cluster 2 – Críticos (Desanimados)**
  - Más del 58% desanimados, casi sin florecidos.  
  - Este cluster concentra los casos de **mayor afectación clínica** y debería recibir atención prioritaria.

---

## 
La curva del codo (**N_Clusters.py**) sugería k=2, pero este valor **diluye a los florecidos** casi por completo.  

Con k=3 se logra un **balance clínicamente más útil**, diferenciando:
  1. Grupo crítico (alta proporción de desanimados).  
  2. Grupo intermedio (moderados con cierto riesgo).  
  3. Grupo positivo (moderados con mayor proporción de florecidos).  

---

## Conclusión
El modelo con **k=3** ofrece una estructura clara y aplicable en el ámbito clínico:

- **Cluster 2** identifica con precisión a la población **en mayor riesgo**.  
- **Clusters 0 y 1** distinguen entre estudiantes con **funcionamiento positivo** y aquellos en un **estado intermedio**, donde se pueden implementar estrategias de prevención.  

Este resultado puede servir como **base inicial para intervenciones diferenciadas** en salud mental, reconociendo que se requiere mayor refinamiento, validación con nuevas muestras y estudios longitudinales para su aplicación definitiva.

---

## Ajustes y optimización

### Versión Original

```python
kmeans = KMeans(
    n_clusters=3, 
    max_iter=1500, 
    tol=1e-4, 
    random_state=0, 
    n_init=50
)
```

### Resultados:

* **Distribución de `mhc_dx` por cluster (proporciones):**

  ```
  cluster   0        1        2
  0       0.1916   0.7584   0.0499
  1       0.5812   0.4109   0.0078
  2       0.1146   0.7606   0.1248
  ```

* **Conteo de `mhc_dx` por cluster:**

  ```
  cluster   0     1     2
  0       541  2141   141
  1      1413   999    19
  2       294  1951   320
  ```

### Versión Ajustada

```python
kmeans = KMeans(
    n_clusters=3, 
    max_iter=300, 
    tol=1e-5, 
    random_state=0, 
    init="k-means++"
)
```

### Resultados:

* **Distribución de `mhc_dx` por cluster (proporciones):**

  ```
  cluster   0        1        2
  0       0.5803   0.4124   0.0074
  1       0.1871   0.7610   0.0519
  2       0.1189   0.7579   0.1232
  ```

* **Conteo de `mhc_dx` por cluster:**

  ```
  cluster   0     1     2
  0      1417  1007    18
  1       526  2140   146
  2       305  1944   316
  ```

### Cambios Realizados

1. **Iteraciones máximas (`max_iter`)**

   * Antes: `1500`
   * Ahora: `300`

     **Justificación:** se redujo para evitar exceso de iteraciones innecesarias, dado que la convergencia era estable con menos pasos.

2. **Tolerancia (`tol`)**

   * Antes: `1e-4`
   * Ahora: `1e-5`
    
      **Justificación:** criterio de convergencia más estricto, buscando una partición más precisa.

3. **Inicialización (`init`)**

   * Antes: inicialización aleatoria (por defecto con `n_init=50`).
   * Ahora: `k-means++`.

    **n_init** indica que se ejecuta el algoritmo 50 veces con inicializaciones aleatorias diferentes de los centroides y elige el resultado con menor inercia (menor suma de distancias al cuadrado).

    **k-means++** es un método inteligente de inicialización que elige los centroides iniciales de forma estratégica, no aleatoria. Básicamente selecciona el primer centro al azar y luego selecciona los siguientes centroides probabilísticamente, favoreciendo puntos más distantes entre sí.

     **Justificación:** mejora la estabilidad inicial de los centroides, reduciendo la variabilidad entre ejecuciones.

### Análisis de Impacto

* **Cluster 0**

  * Original: representaba mayormente clase `1` (75.8%), con algo de `0`.
  * Ajustado: ahora es el cluster más asociado con clase `0` (58%), con mezcla de `1`.
  * **Cambio:** el cluster 0 pasó a segmentar mejor la clase `0`.

* **Cluster 1**

  * Original: fuerte asociación con clase `1` (76.1%).
  * Ajustado: mantiene asociación con clase `1`, pero captura más clase `2` (5.2% frente a 4.9%).
  * **Cambio:** ligero aumento en representación de la clase minoritaria.

* **Cluster 2**

  * Original: era el cluster que más agrupaba clase `2` (12.5%).
  * Ajustado: mantiene ese rol (12.3%), con resultados muy similares.
  * **Cambio:** estabilidad en detección de la clase `2`.

## Conclusiones

1. **Mejora en estabilidad**: el uso de `k-means++` dio una asignación más consistente de los clusters, reduciendo aleatoriedad.
2. **Cluster 0 se redefinió**: pasó de estar dominado por la clase `1` a estar más ligado con la clase `0`, mejorando la segmentación.
3. **Clase minoritaria (2)**: se mantiene mejor representada en el cluster 2, sin deterioro respecto a la versión original.
4. **Eficiencia**: con menos iteraciones (`300` vs `1500`), se logra convergencia más rápida sin pérdida significativa en resultados.

---