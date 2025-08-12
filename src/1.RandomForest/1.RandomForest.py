import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score

# --- 1. Cargar data ---
df = pd.read_csv("data/4datasetListo.csv")

# --- 2. Selección de features y target ---
# Renombrar primero
df = df.rename(columns={
    'Tabaco¿Quétipodesustanciaspsicoactivashaconsumidoenlosúltimos': 'Tabaco',
    'Marihuana¿Quétipodesustanciaspsicoactivashaconsumidoenlosúlti': 'Marihuana',
    'Alcohol¿Quétipodesustanciaspsicoactivashaconsumidoenlosúltimo': 'Alcohol'
})

# Definir las features ya con los nombres nuevos
features = [
    'SUMPHQ', 'SumaGAD', 'SUMCDrisc',  # escalas
    'edad', 'Trabajo', 'Sexo', 'Religion', 'Etnia', 'EstCivil',
    'Terapia', 'TrataPsi', '¿Tienehijos', 'UnAca', 'CEntroU','ConsumoSustancias',
    'Grado', 'Semestre', 'Jornada', 'Tabaco', 'Marihuana', 'Alcohol'
]

target = 'mhc_dx'

X = df[features]
y = df[target]

# --- 3. Dividir en train/test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Cambia aquí: usa DecisionTreeClassifier o RandomForestClassifier
use_random_forest = True

if use_random_forest:
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
else:
    model = DecisionTreeClassifier(random_state=42)

# --- 4. Entrenar ---
model.fit(X_train, y_train)

# --- 5. Predicciones ---
y_pred = model.predict(X_test)

# --- 6. Evaluar ---
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
print("Macro F1:", f1_score(y_test, y_pred, average='macro'))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred, digits=3))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

# --- 7. Validación cruzada (solo accuracy) ---
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("\nAccuracy CV (5 folds):", cv_scores.mean(), "±", cv_scores.std())

# --- 8. Importancia de features ---
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\nImportancia de features:\n", importances)

# Gráfico de importancia
plt.figure(figsize=(8, 5))
importances.plot(kind='barh')
plt.title("Importancia de variables")
plt.gca().invert_yaxis()
plt.show()

# --- 9. Reglas del árbol si es DecisionTree ---
if not use_random_forest:
    tree_rules = export_text(model, feature_names=features)
    # print("\nReglas del árbol:\n", tree_rules)

# --- 10. Guardar modelo ---
#joblib.dump(model, "models/tree_or_forest_model.joblib")
