# Imports necesarios
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("./articulos_ml.csv")
#veamos cuantas dimensiones y registros contiene
print(data.shape)
#son 161 registros con 8 columnas. Veamos los primeros registros
print(data.head())
# Ahora veamos algunas estadísticas de nuestros datos
print(data.describe())

# Visualizamos rápidamente las caraterísticas de entrada
data.drop(['Title', 'url', 'Elapsed days'], axis=1).hist()
plt.show()

#Vamos a RECORTAR los datos en la zona donde se concentran más los puntos
#esto es en el eje X: entre0y3.500
#y en el eje Y:entre 0 y80.000
filtered_data=data[(data['Word count']<=3500)&(data['# Shares']<=80000)]

colores=['orange','blue']
tamanios=[30,60]

f1=filtered_data['Word count'].values
f2=filtered_data['# Shares'].values

#Vamos a pintar en colores los puntos por debajo y por encima de la media d eCantidad de Palabras
asignar=[]
for index, row in filtered_data.iterrows():
    if row['Word count']>1808:
        asignar.append(colores[0])
    else:
        asignar.append(colores[1])

plt.scatter(f1,f2,c=asignar,s=tamanios[0])
plt.show()

# Asignamos nuestra variable de entrada X para entrenamiento y las etiquetas Y.
dataX = filtered_data[["Word count"]]
X_train = np.array(dataX)
y_train = filtered_data['# Shares'].values
# Creamos el objeto de Regresión Linear
regr = linear_model.LinearRegression()
# Entrenamos nuestro modelo
regr.fit(X_train, y_train)
# Hacemos las predicciones que en definitiva una línea (en este caso, al ser 2D)
y_pred = regr.predict(X_train)
# Veamos los coeficientes obtenidos,En nuestro caso, serán la Tangente
print('Coefficients:\n', regr.coef_)
# Este es el valor donde corta el ejeY(enX=0)
print('Independent term:\n', regr.intercept_)
# ErrorCuadradoMedio
print("Mean squared error:%.2f" % mean_squared_error(y_train, y_pred))
# Puntaje de Varianza. El mejor puntaje es un 1.0
print('Variance score:%.2f' % r2_score(y_train, y_pred))

plt.scatter(X_train[:,0], y_train,  c=asignar, s=tamanios[0])
plt.plot(X_train[:,0], y_pred, color='red', linewidth=3)

plt.xlabel('Cantidad de Palabras')
plt.ylabel('Compartido en Redes')
plt.title('Regresión Lineal')

plt.show()

#Vamos a comprobar:
# Quiero predecir cuántos "Shares" voy a obtener por un artículo con 2.000 palabras,
# según nuestro modelo, hacemos:
y_Dosmil = regr.predict([[2000]])
print(int(y_Dosmil[0]))