import joblib
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# --- 3. Dividir en train/test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 4. Escalado ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 5. Entrenamiento del modelo ---
model = MLPClassifier(
    hidden_layer_sizes=(30,20),
    max_iter=3000,
    random_state=42,
    alpha=0.01,
    activation='logistic'
)
model.fit(X_train_scaled, y_train)

# --- 6. Predicciones y evaluación ---
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo:", accuracy)

# --- 7. Reporte detallado ---
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# guardar el modelo completo
#joblib.dump(model, "models/MLP/MLP_classifier.joblib")