import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import StandardScaler

# Cargar datos
dataframe = pd.read_csv(r"usuarios_win_mac_lin.csv")
print(dataframe.head(), "\n")
print(dataframe.describe(), "\n")
print(dataframe.groupby('clase').size())

# Histogramas
dataframe.drop(['clase'], axis=1).hist()
plt.show()

# Pairplot con Seaborn
sb.pairplot(dataframe.dropna(), hue='clase', height=4, vars=["duracion", "paginas", "acciones", "valor"], kind='reg')
plt.show()

# Definir X e y
X = dataframe.drop(['clase'], axis=1).values  # 🔹 Convertir a ndarray
y = dataframe['clase'].values

# 🚀 Normalizar los datos para mejorar la convergencia
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Entrenar modelo con iteraciones
model = linear_model.LogisticRegression(max_iter=1000)
model.fit(X, y)

predictions = model.predict(X)
# Predicciones iniciales
print("Primeras 5 predicciones:", predictions[:5])
print("Precisión en entrenamiento:", model.score(X, y))


# División de datos en entrenamiento y validación
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)

# Validación cruzada
kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=seed)
cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
print(f"LogisticRegression: {cv_results.mean():.4f} ({cv_results.std():.4f})")

# Predicción en datos de validación
predictions = model.predict(X_validation)
print("Precisión en validación:", accuracy_score(Y_validation, predictions))
print("Matriz de confusión:\n", confusion_matrix(Y_validation, predictions))
print("Reporte de clasificación:\n", classification_report(Y_validation, predictions))

# Predicción con un nuevo dato
X_new = np.array([[10, 3, 5, 9]])  # 🔹 Convertir a ndarray
X_new = scaler.transform(X_new)  # 🔹 Aplicar la misma normalización
new_prediction = model.predict(X_new)
print("Predicción para el nuevo usuario:", new_prediction)
