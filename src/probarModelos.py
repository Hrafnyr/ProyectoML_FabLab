import joblib
import pandas as pd
loaded_pipeline = joblib.load('models/gnb_pipeline.joblib')

# _________________ Para modelo 1.1
import numpy as np

# Rangos máximos
max_vals = {
    'SUMPHQ': 27,
    'SumaGAD': 21,
    'SUMCDrisc': 40,
    'mhc_total': 70
}

# Listas de categorías
trabajo_vals = [0, 1, 2]  # jornada completa, medio tiempo, emprendimiento
sexo_vals = [1, 2, 3]     # hombre, mujer, prefiero no decirlo
consumo_vals = [0, 1]     # no, sí

# Generar 10 ejemplos aleatorios variados
np.random.seed(42)
data = []
for _ in range(10):
    ejemplo = {
        'SUMPHQ': np.random.randint(0, max_vals['SUMPHQ'] + 1),
        'SumaGAD': np.random.randint(0, max_vals['SumaGAD'] + 1),
        'SUMCDrisc': np.random.randint(0, max_vals['SUMCDrisc'] + 1),
        'mhc_total': np.random.randint(0, max_vals['mhc_total'] + 1),
        'edad': np.random.randint(18, 65),
        'Trabajo': np.random.choice(trabajo_vals),
        'Sexo': np.random.choice(sexo_vals),
        'ConsumoSustancias': np.random.choice(consumo_vals)
    }
    data.append(ejemplo)

df_test = pd.DataFrame(data)
#print(df_test)


# Predecir
prediccion = loaded_pipeline.predict(df_test)
probas = loaded_pipeline.predict_proba(df_test)  # probabilidades por clase si quieres confianza
classes = loaded_pipeline.named_steps['gaussiannb'].classes_


print("Predicción de clase:", prediccion)
print("Probabilidades:", probas)
# _____________________________________