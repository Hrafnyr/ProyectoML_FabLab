import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline  # <-- para pipeline integrado

# --- 1. Cargar dataset ---
df = pd.read_csv("data/4datasetListo.csv")

# --- 2. Definir features y target ---
features = [
    'SUMPHQ', 'SumaGAD', 'SUMCDrisc',  # escalas
    'edad', 'Semestre','UnAca', 'Trabajo', 'Religion', 'EstCivil', #Demográficas
    'CEntroU','Jornada'
]
target = 'mhc_dx'

X = df[features]
y = df[target]

# --- 3. División train/test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 4. Crear pipeline: escalado + KNN ---
pipeline = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5, weights='distance')
)

# --- 5. Entrenar modelo con pipeline ---
pipeline.fit(X_train, y_train)

# --- 6. Predicción y métricas ---
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nMétricas de evaluación:")
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

# --- 7. Validación cruzada ---
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
print(f"\nAccuracy CV (5 folds): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# --- 8. Visualización simple con dos features ---
# Nota: solo 2 primeras features para ilustrar, KNN usa todas.
# X_vis = StandardScaler().fit_transform(X_train)[:, :2]
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

# --- 9. Guardar pipeline completo ---
joblib.dump(pipeline, "models/KNN/KNN_model.joblib")
