import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import tree

# --- 1. Cargar data ---
df = pd.read_csv("data/4datasetListo.csv")

# --- 2. Selección de features y target ---
features = [
    'SUMPHQ', 'SumaGAD', 'SUMCDrisc', 'mhc_total',  # escalas
    'edad', 'Trabajo', 'Sexo', 'ConsumoSustancias'  # demográficas / contexto
]
target = 'mhc_dx'  # bienestar categórico

X = df[features]
y = df[target]

# --- 3. Dividir en train/test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 4. Entrenar Decision Tree ---
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# --- 5. Predecir ---
y_pred = model.predict(X_test)

# --- 6. Evaluar ---
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred, digits=3))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

# --- 7. Importancia de features ---
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\nImportancia de features:\n", importances)

# --- 8. Reglas del árbol (texto) ---
tree_rules = export_text(model, feature_names=features)
#print("\nReglas del árbol:\n", tree_rules)

# guardar el modelo completo
joblib.dump(model, "models/decision_tree_model.joblib")