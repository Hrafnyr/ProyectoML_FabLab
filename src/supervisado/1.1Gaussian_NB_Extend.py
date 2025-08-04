# Importaciones
import pandas as pd
import joblib  # para guardar/recuperar
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Leer el archivo CSV
df = pd.read_csv('data/4datasetListo.csv')

# --- 2. Selección de features y target ---
features = [
    'SUMPHQ', 'SumaGAD', 'SUMCDrisc', 'mhc_total',  # escalas
    'edad', 'Trabajo', 'Sexo', 'ConsumoSustancias', 'Semestre', 'EstCivil'  # demográficas / contexto social
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

# --- Métricas ---
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=3))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# --- 5. Guardar el pipeline entrenado ---
#joblib.dump(pipeline, 'gnb_pipeline.joblib')  # guarda scaler + modelo en un solo objeto

