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
  Se definen 5 clusters (`n_clusters=5`) y se ajusta el modelo a los datos escalados.

- **Predicción y análisis:**  
  Se asigna a cada individuo un cluster y se analiza la distribución relativa (`proporciones`) y absoluta (`conteos`) de la variable categórica `mhc_dx` en cada cluster.

- **Visualización:**  
  Se genera un gráfico de barras apiladas mostrando la proporción de cada categoría `mhc_dx` dentro de cada cluster, para facilitar la interpretación.

- **Guardado de modelos:**  
  Tanto el modelo KMeans como el scaler se guardan para su uso futuro en inferencia.

---

## Resultados del Análisis

### Distribución de `mhc_dx` por Cluster (Proporciones)

| cluster | 0 (Desanimado) | 1 (Moderado) | 2 (Florecido) |
|---------|----------------|--------------|---------------|
| 0       | 17.4%          | 82.3%        | 0.3%          |
| 1       | 84.9%          | 15.0%        | NaN           |
| 4       | 1.0%           | 73.8%        | 25.2%         |

### Conteo Absoluto de `mhc_dx` por Cluster

| cluster | 0     | 1     | 2    |
|---------|-------|-------|------|
| 0       | 167   | 788   | 3    |
| 1       | 1514  | 268   | NaN  |
| 2       | 282   | 2196  | 3    |
| 3       | 267   | 478   | 9    |
| 4       | 18    | 1361  | 465  |

---

## Interpretación

- Los clusters revelan agrupamientos con predominancia de ciertas categorías de bienestar mental: por ejemplo, el cluster 1 concentra principalmente individuos desanimados (clase 0), mientras que el cluster 0 agrupa mayormente individuos en estado moderado (clase 1).
- El cluster 4 muestra una mezcla más equilibrada, incluyendo una proporción significativa de individuos florecidos (clase 2).
- Esto sugiere que el agrupamiento no supervisado puede identificar perfiles o subpoblaciones con características clínicas y sociodemográficas similares, que se relacionan con distintos estados de bienestar.

---

## Ejemplo de Uso Práctico

```plaintext
Predicción de clase: [0 0 1 1 2 1 2 2 2 2]
Probabilidades: [
  [1. 0. 0.],
  [1. 0. 0.],
  [0. 1. 0.],
  [0. 1. 0.],
  [0. 0. 1.],
  [0. 1. 0.],
  [0. 0. 1.],
  [0. 0. 1.],
  [0. 0. 1.],
  [0. 0. 1.]
]
```

Para esta prueba se usó un individuo con valores aleatorios y el modelo lo asignó al cluster 4.

### Interpretación de la asignación al **Cluster 4**

Cuando el modelo KMeans asigna a un individuo al **cluster 4**, significa que, en base a sus características clínicas, demográficas y sociales, este individuo pertenece a un grupo (cluster) específico dentro de la población analizada que comparte patrones similares.

---

#### ¿Qué caracteriza al **cluster 4**?

Según el análisis previo de la distribución de bienestar mental (`mhc_dx`) dentro de los clusters:

- **Cluster 4** está compuesto mayormente por personas con bienestar moderado (`mhc_dx = 1`) en aproximadamente un **73.8%** de los casos.
- Además, tiene una proporción significativa (alrededor del **25.2%**) de personas en estado floreciente (`mhc_dx = 2`), lo cual indica una buena salud mental.
- Solo una pequeña minoría corresponde a individuos desanimados (`mhc_dx = 0`), alrededor del **1%**.

Esto indica que el cluster 4 agrupa a individuos con un perfil más positivo en cuanto a bienestar mental, con alta probabilidad de encontrarse en estados moderados o florecientes.

---

#### ¿Qué implica para el nuevo individuo?

Asignar al nuevo individuo al cluster 4 implica que sus características se parecen más a este grupo saludable/moderado dentro del conjunto de datos, por lo que:

- Probablemente presenta un bienestar mental adecuado o positivo.
- Comparte características similares en las variables clínicas (como puntajes de escalas) y demográficas con otros individuos de este cluster.
- Podría beneficiarse o comportarse de forma parecida a la mayoría de personas de este cluster en términos de intervenciones, seguimiento o análisis clínico.

Este análisis permite interpretar las agrupaciones.