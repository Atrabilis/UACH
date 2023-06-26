import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.initializers import he_uniform, he_normal

# Limpia la consola
os.system("cls")

# Ruta del archivo CSV de entrenamiento
ruta_entrenamiento_csv = os.path.join(os.getcwd(), 'entrenamiento.csv')

# Ruta del archivo CSV de prueba
ruta_prueba_csv = os.path.join(os.getcwd(), 'prueba.csv')

# Leer el archivo CSV de entrenamiento y cargar los datos y etiquetas
datos_entrenamiento = []
with open(ruta_entrenamiento_csv, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        datos_entrenamiento.append(row)

# Leer el archivo CSV de prueba y cargar los datos y etiquetas
datos_prueba = []
with open(ruta_prueba_csv, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        datos_prueba.append(row)

# Preparar los datos y etiquetas de entrenamiento
X_train = []
y_train = []
for dato in datos_entrenamiento:
    ruta_imagen = dato['Ruta']
    etiqueta = int(dato['Etiqueta'])
    
    # Cargar la imagen y convertirla a escala de grises
    imagen = Image.open(ruta_imagen).convert('L')
    
    # Cambiar el tamaño de la imagen a 328x364
    imagen = imagen.resize((328, 364))
    
    # Convertir la imagen a un array de numpy
    imagen = np.array(imagen)
    
    X_train.append(imagen)
    y_train.append(etiqueta)

# Preparar los datos y etiquetas de prueba
X_test = []
y_test = []
for dato in datos_prueba:
    ruta_imagen = dato['Ruta']
    etiqueta = int(dato['Etiqueta'])
    
    # Cargar la imagen y convertirla a escala de grises
    imagen = Image.open(ruta_imagen).convert('L')
    
    # Cambiar el tamaño de la imagen a 328x364
    imagen = imagen.resize((328, 364))
    
    # Convertir la imagen a un array de numpy
    imagen = np.array(imagen)
    
    X_test.append(imagen)
    y_test.append(etiqueta)

# Convertir los datos y etiquetas a arrays de numpy
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Normalizar los valores de píxeles en el rango [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Crear el modelo de red neuronal
model = Sequential()
model.add(Flatten(input_shape=(364, 328)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compilar el modelo
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Definir el porcentaje objetivo de precisión
porcentaje_objetivo = 0.99

# Entrenar el modelo hasta alcanzar el porcentaje objetivo de precisión
epocas = 0
while True:
    history = model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=2)
    epocas += 1
    _, accuracy = model.evaluate(X_test, y_test)
    if accuracy >= porcentaje_objetivo:
        break

# Imprimir el número de épocas necesarias para alcanzar el porcentaje objetivo
print('Épocas requeridas:', epocas)

# Evaluar el modelo en los datos de prueba
loss, accuracy = model.evaluate(X_test, y_test)
print('Pérdida en los datos de prueba:', loss)
print('Precisión en los datos de prueba:', accuracy)

# Graficar la precisión durante el entrenamiento
plt.plot(history.history['accuracy'])
plt.title('Precisión del modelo')
plt.ylabel('Precisión')
plt.xlabel('Época')
plt.legend(['Entrenamiento', 'Prueba'], loc='upper left')
plt.show()
