# Random Forest

El algoritmo "Random Forest" permite evaluar múltiples árboles de decisión basado en los features/variables hacie una variable objetivo/target (Y).

Esta forma paralela de evaluar multiples árboles tomando de manera aleatoria cada feature permite obtener una base suficientemente sólida para poder identificar **las variables de mayor impacto** en los datos. De este modo pueden identificar cuales no aportan información para el entrenamiento y cuales sí.

# Análisis de Resultados
![RandomForest](../img/RandomForestResults.png)

¡Muy buen resultado para ser un primer modelo! Aquí un resumen rápido para interpretar lo que ves:

---

### Métricas principales

* **Accuracy \~74.3%:** El porcentaje total de aciertos está bien, pero en clases desbalanceadas puede ser engañoso.
* **Balanced Accuracy \~51%:** Corrige el desbalance de clases; indica que el modelo no está tan bueno en todas las clases.
* **Macro F1 \~53%:** Promedio de F1 por clase, muestra que en general el desempeño es moderado, con clases más difíciles.

---

### Por clase

* Clase **0 (Desanimado)**: precisión decente (0.69), pero recall bajo (0.56), muchas predicciones faltaron (falsos negativos).
* Clase **1 (Moderado)**: excelente recall (0.89) y buena precisión (0.76), se predice muy bien.
* Clase **2 (Florecido)**: precisión alta (0.80) pero recall bajísimo (0.08), casi no detecta a los florecidos correctamente (falsos negativos muy altos).

---

### Matriz de confusión

* Muchos florecidos clasificados erróneamente como moderados (87 casos).
* Modelo tiende a confundir clases menos frecuentes con la clase mayoritaria/moderada.

---

### Importancia de features

* Variables clave:

  * `SUMCDrisc`, `SUMPHQ`, `SumaGAD` (tests o escalas psicológicas), que tienen mucho peso — tiene sentido, pues son indicadores directos.
  * `edad` también influye bastante.
* Variables demográficas y consumo tienen menor relevancia (pero no descartes que puedan aportar).

---

### En resumen

* El modelo funciona bien para la clase **moderado** (la mayoría), pero le cuesta distinguir bien las clases minoritarias, sobre todo "Florecido".


---
## Valoración de variables candidatas

Se tomarán las primeras 11 características:
|Nombre|Valor|
|--|--|
 |SUMCDrisc    |        0.208582|
 |SUMPHQ       |        0.179353|
 |SumaGAD       |       0.115870|
 |edad           |      0.082559|
 |Semestre        |     0.057333|
 |UnAca            |    0.054795|
 |Religion          |   0.039205|
|Jornada            |  0.037379|
 |CEntroU             | 0.036508|
| Trabajo              |0.028360|
| EstCivil             |0.026593|

Se agrega el uso del class_weight: esto permite mejorar ligeramente el recall de la clase minoritaria (aunque sigue bajo), y en la validación cruzada el promedio sube (0.56 → 0.60). Esto significa que está ayudando a que el modelo sea un poco más sensible a las clases pequeñas.