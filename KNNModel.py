
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline  # <-- para pipeline integrado
from pathlib import Path

def crear_model_KNN(file_csv='4datasetListo.csv', file_model='m3_KNN_model.joblib'):

    file_model = Path(file_model)
    if file_model.exists():
        # print(f"El modelo ya existe: {file_model}")
        return file_model

    # --- 1. Cargar dataset ---
    df = pd.read_csv(file_csv)

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

    # --- 7. Validación cruzada ---
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    #print(f"\nAccuracy CV (5 folds): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # --- 9. Guardar pipeline completo ---
    joblib.dump(pipeline, file_model)
    return file_model