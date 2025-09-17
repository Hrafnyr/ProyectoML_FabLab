# Importaciones
import pandas as pd
import joblib  # para guardar/recuperar
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from pathlib import Path

def crear_modelo_GNB(file_csv='4datasetListo.csv', file_model='m1_GaussianNB.joblib'):

    #Validar si el modelo existe
    file_model = Path(file_model)
    if file_model.exists():
        #print(f"El modelo ya existe: {file_model}")
        return file_model

    # Leer el archivo CSV
    df = pd.read_csv(file_csv)

    # --- 2. Selección de features y target ---
    features = [
        'SUMPHQ', 'SumaGAD', 'SUMCDrisc',  # escalas
        'edad', 'Semestre','UnAca', 'Trabajo', 'Religion', 'EstCivil', #Demográficas
        'CEntroU','Jornada'
    ]

    target = 'mhc_dx'  # ejemplo: bienestar categórico (0=languishing,1=moderado,2=floreciente)

    X = df[features]
    y = df[target]

    # --- 3. Dividir en train/test ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Pipeline: escalado + GaussianNB ---
    pipeline = make_pipeline(StandardScaler(), GaussianNB())
    pipeline.fit(X_train, y_train)

    # Definir validación cruzada estratificada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Evaluar con cross_val_score usando accuracy
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

    # --- Métricas ---
    y_pred = pipeline.predict(X_test)

    # --- 5. Guardar el pipeline entrenado ---
    joblib.dump(pipeline, file_model)  # guarda scaler + modelo en un solo objeto
    return file_model

