import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

def crear_model_Kmeans(file_csv='4datasetListo.csv', file_model='m2_modelo_kmeans.pkl', file_scaler='m2_scaler.pkl'):

    file_model = Path(file_model)
    if file_model.exists():
        # print(f"El modelo ya existe: {file_model}")
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

    # --- 3. Escalar los datos ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 4. Entrenar modelo KMewans ---
    kmeans = KMeans(n_clusters=3, max_iter=1500, tol=1e-4, random_state=0,n_init=50)
    kmeans.fit(X_scaled)

    # --- 5. Predicción de los clusters ---
    y_predict = kmeans.predict(X_scaled)

    # --- 6. Obtener los centroides ---
    centroids = kmeans.cluster_centers_

    # Agregar la predicción del cluster al DataFrame original
    df['cluster'] = y_predict


    #importar y guardar
    joblib.dump(kmeans, file_model)

    # Guardar el scaler
    joblib.dump(scaler, file_scaler)