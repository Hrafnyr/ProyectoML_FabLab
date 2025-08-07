import joblib
import pandas as pd
loaded_pipeline = joblib.load('models/gnb_pipeline.joblib')
loades_DecisionTree = joblib.load('models/decision_tree_model.joblib')


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


#____________________ Predecir gaussian NB
prediccion = loaded_pipeline.predict(df_test)
probas = loaded_pipeline.predict_proba(df_test)  # probabilidades por clase si quieres confianza
classes = loaded_pipeline.named_steps['gaussiannb'].classes_


print("Predicción de clase:", prediccion)
print("Probabilidades:", probas)
#____________________________________

#____________________ Predecir Decision tree
prediccion = loades_DecisionTree.predict(df_test)
probas = loades_DecisionTree.predict_proba(df_test)  # probabilidades por clase si quieres confianza

print("Predicción de clase:", prediccion)
print("Probabilidades:", probas)
#____________________________________

#______________________ KMEANS _____________ 
# Cargar modelo y scaler
kmeans = joblib.load('models/KMEANS/modelo_kmeans.pkl')
scaler = joblib.load('models/KMEANS/scaler.pkl')

# Crear un DataFrame con los datos de 1 persona (deben estar en el mismo orden que tus 'features')
nuevo_dato = pd.DataFrame([{
    'SUMPHQ': 23,
    'SumaGAD': 12,
    'SUMCDrisc': 7,
    'mhc_total': 48,
    'mhc_ewb': 16,
    'loaff': 2,
    'hiaffect': 5,
    'edad': 22,
    'Sexo': 1,
    'Trabajo': 1,
    'Religion': 1,
    'ConsumoSustancias': 0,
    'Semestre': 4,
    'EstCivil': 0,
    'Terapia': 1,
    'TrataPsi': 0,
    'UnAca': 1,
    'Grado': 0
}])

# Escalar los datos igual que como se hizo en entrenamiento
nuevo_dato_scaled = scaler.transform(nuevo_dato)

# Predecir el cluster
cluster_predicho = kmeans.predict(nuevo_dato_scaled)

print(f"El individuo fue asignado al cluster: {cluster_predicho[0]}")
