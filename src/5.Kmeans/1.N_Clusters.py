import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

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

# --- Búsqueda del mejor número de clusters ---
inercia = []
silhouette_scores = []
K_range = range(2, 11)  # probar de 2 a 10 clusters

for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, max_iter=300, tol=1e-4, random_state=0)
    kmeans_temp.fit(X_scaled)
    inercia.append(kmeans_temp.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans_temp.labels_))

# --- Gráfica Elbow Method ---
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(K_range, inercia, marker='o')
plt.title('Método del Codo')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Inercia')
plt.grid(True)

# --- Gráfica Silhouette Score ---
plt.subplot(1,2,2)
plt.plot(K_range, silhouette_scores, marker='o', color='orange')
plt.title('Puntaje de Silhouette')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)

plt.show()

# Elegir k óptimo
mejor_k = K_range[silhouette_scores.index(max(silhouette_scores))]
print(f"✅ Mejor número de clusters según Silhouette: {mejor_k}")
