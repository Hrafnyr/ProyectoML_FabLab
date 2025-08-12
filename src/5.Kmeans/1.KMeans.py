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
    'SUMPHQ', 'SumaGAD', 'SUMCDrisc', 'mhc_total', 'mhc_ewb','loaff', 'hiaffect',  # escalas
    'edad', 'Sexo', 'Trabajo','Religion','ConsumoSustancias', 'Semestre',
    'EstCivil', 'Terapia', 'TrataPsi', 'UnAca','Grado'  # demográficas / contexto social
]
target = 'mhc_dx'  # bienestar categórico

X = df[features]
y = df[target]

# --- 3. Escalar los datos ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 4. Entrenar modelo KMeans ---
kmeans = KMeans(n_clusters=5, max_iter=300, tol=1e-4, random_state=0)
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


# # --- 8. Visualización 2D (usando las dos primeras features escaladas) ---
# plt.figure(figsize=(8, 6))

# #Suficientes colores para la cantidad de clusters
# colors = ['red', 'green', 'blue', 'purple', 'orange', 'cyan', 'magenta']

# # # Usar solo las dos primeras columnas de X_scaled para graficar
# x_axis = X_scaled[:, 0]
# y_axis = X_scaled[:, 1]

# # Dibujar cada punto con el color de su cluster asignado
# for i in range(len(X_scaled)):
#     plt.scatter(x_axis[i], y_axis[i], color=colors[y_predict[i]], s=50)

# # Dibujar centroides
# for i in range(len(centroids)):
#     plt.scatter(centroids[i, 0], centroids[i, 1], color='black', s=200, marker='X', label=f'Centroid {i+1}')

# plt.title("Clustering con KMeans (2 features)")
# plt.xlabel("Feature 1: SUMPHQ (escalado)")
# plt.ylabel("Feature 2: SumaGAD (escalado)")
# plt.legend()
# plt.grid(True)
# plt.show()

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