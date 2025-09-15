import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# --- 1. Cargar data ---
df = pd.read_csv("data/4datasetListo.csv")

# --- 2. Selección de features ---
df = df.rename(columns={
    'Tabaco¿Quétipodesustanciaspsicoactivashaconsumidoenlosúltimos': 'Tabaco',
    'Marihuana¿Quétipodesustanciaspsicoactivashaconsumidoenlosúlti': 'Marihuana',
    'Alcohol¿Quétipodesustanciaspsicoactivashaconsumidoenlosúltimo': 'Alcohol'
})

features = [
    'SUMPHQ', 'SumaGAD', 'SUMCDrisc',  # escalas
    'edad', 'Trabajo', 'Sexo', 'Religion', 'Etnia', 'EstCivil',
    'Terapia', 'TrataPsi', '¿Tienehijos', 'UnAca', 'CEntroU','ConsumoSustancias',
    'Grado', 'Semestre', 'Jornada', 'Tabaco', 'Marihuana', 'Alcohol'
]

X = df[features]

# --- 3. Escalar datos ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 4. Aplicar PCA ---
pca = PCA(n_components=0.95, random_state=42)  # Mantener 95% de la varianza
X_pca = pca.fit_transform(X_scaled)

# --- 5. Resultados ---
print("Shape original:", X.shape)
print("Shape después de PCA:", X_pca.shape)

explained_var = pca.explained_variance_ratio_
print("\nVarianza explicada por cada componente:\n", explained_var)
print("\nVarianza acumulada:\n", explained_var.cumsum())

# --- 6. Gráfico de varianza explicada ---
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_var) + 1), explained_var.cumsum(), marker='o')
plt.xlabel("Número de Componentes")
plt.ylabel("Varianza Acumulada Explicada")
plt.title("Curva de varianza explicada por PCA")
plt.grid(True)
plt.show()

# --- 7. Calcular Loadings ---
# Loadings = componentes * sqrt(varianza explicada)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# Convertir a DataFrame para interpretarlo mejor
loadings_df = pd.DataFrame(
    loadings,
    index=features,
    columns=[f"PC{i+1}" for i in range(pca.n_components_)]
)

print("\n--- Loadings (cargas factoriales) ---")
print(loadings_df.round(3))

# Opcional: ver qué variables más pesan en el primer componente
print("\nVariables más influyentes en PC1:")
print(loadings_df["PC1"].abs().sort_values(ascending=False).head(10))
