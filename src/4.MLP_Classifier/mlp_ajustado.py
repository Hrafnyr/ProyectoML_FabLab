import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# --- 1. Cargar data ---
df = pd.read_csv("data/4datasetListo.csv")

# --- 2. Features y target ---
features = [
    'SUMPHQ', 'SumaGAD', 'SUMCDrisc',  # escalas
    'edad', 'Semestre','UnAca', 'Trabajo', 'Religion', 'EstCivil', # demográficas
    'CEntroU','Jornada'
]
target = 'mhc_dx'

X = df[features]
y = df[target]

# --- 3. División en train/test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 4. Pipeline: SMOTE + Escalado + MLP ---
pipeline = Pipeline(steps=[
    # Oversampling
    ('smote', SMOTE(
        sampling_strategy={0: 2500, 2: 1200},
        random_state=42
    )),
    
    # Normalización: importante para MLP
    ('scaler', StandardScaler()),
    
    # Red neuronal
    ('mlp', MLPClassifier(
        hidden_layer_sizes=(100,60),  #  capas y neuronas
        max_iter=1000,               #  nº máximo de iteraciones
        alpha=0.01,                  #  regularización L2 (0.0001, 0.001, 0.1)
        activation='logistic',           #  'relu', 'tanh', 'logistic'
        solver='sgd',               #  'adam' (default), 'sgd', 'lbfgs'
        random_state=42
    ))
])

# --- 5. Entrenamiento ---
pipeline.fit(X_train, y_train)

# --- 6. Predicciones y métricas ---
y_pred = pipeline.predict(X_test)

print("\n🔹 Accuracy:", accuracy_score(y_test, y_pred))
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🔹 Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))

# --- 7. Guardar modelo ---
#joblib.dump(pipeline, "models/MLP/MLP_classifier_smote.joblib")
