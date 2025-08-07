import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- 1. Cargar dataset ---
df = pd.read_csv("data/4datasetListo.csv")

# --- 2. Definir features y target ---
features = [
    'SUMPHQ', 'SumaGAD', 'SUMCDrisc', 'mhc_total', 'mhc_ewb', 'loaff', 'hiaffect',
    'edad', 'Sexo', 'Trabajo', 'Religion', 'ConsumoSustancias', 'Semestre',
    'EstCivil', 'Terapia', 'TrataPsi', 'UnAca', 'Grado'
]
target = 'mhc_dx'

X = df[features]
y = df[target]

# --- 3. División train/test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 4. Escalado ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 5. Definir y entrenar el modelo ---
knn = KNeighborsClassifier(n_neighbors=7, weights='distance')
knn.fit(X_train_scaled, y_train)

# --- 6. Predicción y métricas ---
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("\nMétricas de evaluación:")
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

# # --- 7. Visualización simple con dos features ---
# # Solo para visualización: usamos dos features representativas
# X_vis = X_train_scaled[:, :2]  # Usamos solo las 2 primeras columnas
# y_vis = y_train.to_numpy()

# plt.figure(figsize=(8, 6))
# colores = ['red', 'green', 'blue']
# for i in range(len(X_vis)):
#     plt.scatter(X_vis[i, 0], X_vis[i, 1], color=colores[y_vis[i]], s=50, alpha=0.6)

# plt.title("Visualización KNN (con 2 primeras features)")
# plt.xlabel(features[0])
# plt.ylabel(features[1])
# plt.grid(True)
# plt.show()

# guardar el modelo completo
joblib.dump(knn, "models/KNN/KNN_classifier.joblib")
joblib.dump(scaler, "models/KNN/scaler_knn.joblib")
