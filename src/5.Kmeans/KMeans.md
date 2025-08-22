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
