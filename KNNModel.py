
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.pipeline import Pipeline   # <-- de imblearn
from imblearn.over_sampling import SMOTE
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

    # --- 7. Validación cruzada ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1_macro')

    # --- 9. Guardar pipeline completo ---
    joblib.dump(pipeline, file_model)
    return file_model