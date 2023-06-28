import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

os.system("cls")

# Cargar los datos del archivo CSV
data = pd.read_csv('etiquetas.csv')

# Leer las rutas de las imágenes y las etiquetas
rutas = data['Ruta'].values
etiquetas = data['Etiqueta'].values

# Lista para almacenar las imágenes en forma de matriz
imagenes = []

# Leer las imágenes y convertirlas en matrices
for ruta in rutas:
    imagen = plt.imread(ruta)
    imagenes.append(imagen)

# Convertir las listas en arrays numpy
imagenes = np.array(imagenes)
etiquetas = np.array(etiquetas)

# Dividir los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(imagenes, etiquetas, test_size=0.5, random_state=42)

# Normalizar los valores de píxeles entre 0 y 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Crear el modelo de red neuronal
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(imagen.shape[0], imagen.shape[1])),
    tf.keras.layers.Dense(512, activation='relu'),
    #tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100)

# Mostrar la progresión de la precisión en los datos de entrenamiento y prueba
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Precisión del modelo')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend(['Entrenamiento', 'Prueba'], loc='upper left')
plt.show()
