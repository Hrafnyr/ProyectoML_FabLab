import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- Librerías de Keras ---
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD

# --- 1. Cargar data ---
df = pd.read_csv("4datasetListo.csv")

# --- 2. Selección de features y target ---
features = [
    'SUMPHQ', 'SumaGAD', 'SUMCDrisc',  # escalas
    'edad', 'Semestre','UnAca', 'Trabajo', 'Religion', 'EstCivil', # demográficas
    'CEntroU','Jornada'
]
target = 'mhc_dx'  # bienestar categórico

X = df[features].values
y = df[target].values

# --- 3. Dividir en train/test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 4. Escalado ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 5. Construcción del modelo en Keras ---
model = Sequential()

model.add(Dense(12, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(6, activation='relu'))

model.add(Dense(3, activation='softmax'))

# --- 6. Compilación del modelo ---
# optimizadores: Adam (recomendado), SGD, RMSprop
model.compile(
    loss='sparse_categorical_crossentropy',  # porque y son enteros (0,1,2)
    optimizer=Adam(learning_rate=0.001),     # se puede probar 0.01 o 0.0001
    metrics=['accuracy']
)

# --- 7. Entrenamiento ---
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,     # parte del train para validación
    epochs=150,               # número de iteraciones (aumentar si no converge)
    batch_size=32,            # número de muestras por batch (16, 32, 64)
    verbose=1
)

# --- 8. Evaluación en test ---
y_pred_probs = model.predict(X_test_scaled)
y_pred = y_pred_probs.argmax(axis=1)

accuracy = accuracy_score(y_test, y_pred)
print("\nPrecisión del modelo:", accuracy)

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))


