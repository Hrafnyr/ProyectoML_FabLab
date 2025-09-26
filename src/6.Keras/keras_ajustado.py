import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# --- Librerías de Keras ---
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization

# --- 1. Cargar data ---
df = pd.read_csv("4datasetListo.csv")

# --- 2. Selección de features y target ---
features = [
    'SUMPHQ', 'SumaGAD', 'SUMCDrisc',  
    'edad', 'Semestre','UnAca', 'Trabajo', 'Religion', 'EstCivil', 
    'CEntroU','Jornada'
]
target = 'mhc_dx'  

X = df[features].values
y = df[target].values

# --- 3. División train/test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 4. Escalado ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 5. Balanceo con SMOTE---
smote = SMOTE(sampling_strategy={0:2200, 2:2300}, random_state=42,k_neighbors=10)
X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)

#--- 6. Class weights (opcional si no usas SMOTE) ---
class_weights = compute_class_weight(
   class_weight="balanced",
   classes=np.unique(y_train),
   y=y_train
)
class_weights_dict = dict(enumerate(class_weights))

# --- 7. Construcción del modelo ---
model = Sequential()
model.add(Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))

# --- 8. Compilación ---
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# --- 9. Early stopping ---
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True
)

# --- 10. Entrenamiento ---
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=300,
    batch_size=32,
    callbacks=[early_stop],
    class_weight=class_weights_dict
)

# --- 11. Evaluación ---
y_pred_probs = model.predict(X_test_scaled)
y_pred = y_pred_probs.argmax(axis=1)

accuracy = accuracy_score(y_test, y_pred)
print("\nPrecisión del modelo:", accuracy)

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))