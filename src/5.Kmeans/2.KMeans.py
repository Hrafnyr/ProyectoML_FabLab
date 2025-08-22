import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# --- 1. Cargar data ---
df = pd.read_csv("data/4datasetListo.csv")

# --- 2. Selección de features y target ---
features = [
    'SUMPHQ', 'SumaGAD', 'SUMCDrisc',  # escalas
    'edad', 'Semestre','UnAca', 'Trabajo', 'Religion', 'EstCivil', #Demográficas
    'CEntroU','Jornada'
]

target = 'mhc_dx'  # bienestar categórico

X = df[features]
y = df[target]

# --- 3. Escalar los datos ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 4. Entrenar modelo KMewans ---
kmeans = KMeans(n_clusters=3, max_iter=1500, tol=1e-4, random_state=0,n_init=50)
kmeans.fit(X_scaled)

# --- 5. Predicción de los clusters ---
y_predict = kmeans.predict(X_scaled)

# --- 6. Obtener los centroides ---
centroids = kmeans.cluster_centers_

# --- 7. Mostrar resultados ---
print("\nPredicciones de los clusters (y_predict):")
print(y_predict)

print("\nCentroides de los clusters:")
print(centroids)

# Agregar la predicción del cluster al DataFrame original
df['cluster'] = y_predict

# Mostrar la distribución del target mhc_dx en cada cluster
distribucion = df.groupby('cluster')['mhc_dx'].value_counts(normalize=True).unstack()
print("\nDistribución de 'mhc_dx' por cluster (proporciones):")
print(distribucion)

# También puedes mostrar los conteos absolutos si prefieres
conteos = df.groupby('cluster')['mhc_dx'].value_counts().unstack()
print("\nConteo de 'mhc_dx' por cluster:")
print(conteos)


# Asumiendo que ya tienes df con 'cluster' y 'mhc_dx'
contingencia = pd.crosstab(df['cluster'], df['mhc_dx'], normalize='index')
contingencia.plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Distribución de mhc_dx por Cluster')
plt.xlabel('Cluster')
plt.ylabel('Proporción')
plt.legend(title='mhc_dx', labels=['Desanimado', 'Moderado', 'Florecido'])
plt.show()

#importar y guardar
joblib.dump(kmeans, 'models/KMEANS/modelo_kmeans.pkl')

# Guardar el scaler
joblib.dump(scaler, 'models/KMEANS/scaler.pkl')