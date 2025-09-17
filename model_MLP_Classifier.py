import joblib
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def crear_modelo_MLP(file_csv='4datasetListo.csv', file_model='m4_MLP_classifier.joblib'):

    #Validar si el modelo existe
    file_model = Path(file_model)
    if file_model.exists():
        #print(f"El modelo ya existe: {file_model}")
        return file_model

    # --- 1. Cargar data ---
    df = pd.read_csv(file_csv)

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

    # guardar el modelo completo
    joblib.dump(model, file_model)
    return file_model