# Importaciones
import pandas as pd
import joblib  # para guardar/recuperar
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Leer el archivo CSV
df = pd.read_csv('data/4datasetListo.csv')

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
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=3))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy CV (5 folds): {:.3f} ± {:.3f}".format(cv_scores.mean(), cv_scores.std()))

# --- 5. Guardar el pipeline entrenado ---
joblib.dump(pipeline, 'models/GAUSSIAN_NB/GaussianNB.joblib')  # guarda scaler + modelo en un solo objeto

