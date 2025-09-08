# Selección de características

Como primer punto se evalúan las características.
Para ello usamos RandomForest, si una característica fue tomada en cuenta para definir
límites de decisión, significa que la caracterísitcas es relevante y puede usaer como 
feature. Se agregan números para asignar jeraquía a los archivos de código creados.

# Voting classifier 

El resultado final será entregar un grupo de modelos que puedan predecir o agrupar
correctamente la variable mhc_dx en 0,1,2 (Desanimado, Moderado, Florecido), luego
se escogerá la predicción que más se obtenga de los modelos.

Copyright 2025 Moises David Maldonado de León
Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0
