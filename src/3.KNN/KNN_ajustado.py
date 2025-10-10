import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.pipeline import Pipeline   # <-- de imblearn
from imblearn.over_sampling import SMOTE

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

# --- 4. Pipeline: SMOTE + Escalado + KNN ---
pipeline = Pipeline(steps=[
    ('smote', SMOTE(sampling_strategy={0:2500, 2:1200}, random_state=42)),
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(
        n_neighbors=7, 
        weights='distance',
        metric='minkowski',
        p=2,
        algorithm='auto'
    ))
])

# --- 5. Entrenar modelo ---
pipeline.fit(X_train, y_train)

# --- 6. Predicción y métricas ---
y_pred = pipeline.predict(X_test)
print("\nMétricas de evaluación:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

# --- 7. Validación cruzada (con CV estratificado) ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1_macro')
print(f"\nF1-macro CV (5 folds): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# --- 8. Guardar pipeline ---
joblib.dump(pipeline, "models/KNN/m3_KNN_model.joblib")
