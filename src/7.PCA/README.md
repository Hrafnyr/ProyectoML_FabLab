
## Análisis de PCA

1. **Reducción dimensional**

   * El dataset original tenía **21 variables**.
   * Tras aplicar PCA y conservar el **95% de la varianza**, se redujo a **18 componentes principales**.
   * Esto significa que, aunque hubo una ligera reducción, las variables originales contienen bastante información no redundante.

2. **Varianza explicada**

   * Los primeros **5 componentes** explican casi el **48% de la varianza total**.
   * Con **10 componentes**, se llega al **72%**.
   * Los **18 componentes finales** explican aproximadamente el **96.6%**, lo que garantiza que casi toda la información original está preservada.

3. **Interpretación de los componentes (loadings)**

   * **PC1 (13.4% de varianza):** está fuertemente definido por **Consumo de sustancias (Alcohol, Tabaco, Marihuana, ConsumoSustancias)** y también por síntomas psicológicos (SUMPHQ, SumaGAD). Este componente se interpreta como un **eje de consumo y salud mental**.

   * **PC2 (11.8% de varianza):** se asocia a variables demográficas como **edad, trabajo, tener hijos**, con carga opuesta a escalas psicológicas. Representa un **eje sociodemográfico**.

   * **PC3 (9% de varianza):** combina **sexo y ansiedad (SumaGAD)**, con influencia de nivel académico. Posible **eje académico/psicológico**.

   * Otros componentes posteriores capturan combinaciones más específicas (terapia, centro universitario, grado, etc.), que explican varianza menor pero aportan matices.

---

## Conclusiones

1. **El PCA confirmó que los datos no son altamente redundantes**, ya que se necesitaron 18 de las 21 variables originales para retener el 95% de la información.

2. **Los primeros componentes revelan patrones latentes importantes**:

   * **PC1 (Consumo y salud mental)** muestra que el uso de sustancias y los síntomas psicológicos están fuertemente relacionados y explican gran parte de la variabilidad de los estudiantes.
   * **PC2 (Factores sociodemográficos)** evidencia que edad, trabajo y tener hijos generan otra dimensión distinta de variabilidad.

3. **Desde la perspectiva predictiva (clasificación de mhc\_dx)**, PCA **no mejora el rendimiento**, porque no reduce significativamente la dimensionalidad y Random Forest ya maneja bien la redundancia.

4. **El verdadero aporte del PCA aquí es interpretativo**: permite identificar que el bienestar mental se organiza en al menos dos grandes ejes:

   * Un eje de **consumo + síntomas psicológicos**.
   * Un eje de **factores sociodemográficos**.

5. **Para predicción práctica**, las **11 variables seleccionadas por Random Forest** resultan más útiles que las 18 dimensiones abstractas de PCA, ya que mantienen la interpretabilidad directa de qué factores influyen en el bienestar mental.

---

**En resumen:**

El PCA aporta valor como **herramienta exploratoria y descriptiva**, mostrando las dimensiones subyacentes en los datos. Sin embargo, para la tarea de **clasificación del bienestar mental**, el enfoque con **selección de variables (Random Forest)** es más eficiente y más interpretable.

---