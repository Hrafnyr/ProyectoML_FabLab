import joblib
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
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

    # guardar el modelo completo
    joblib.dump(pipeline, file_model)

    return file_model