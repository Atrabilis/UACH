import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

os.system("cls")

# Cargar el archivo CSV
data = pd.read_csv('labels.csv')

# Obtener las rutas de archivo, etiquetas y nombres de archivo
rutas = data['Ruta'].values
etiquetas = data['Etiqueta'].values
nombres_archivo = data['Archivo'].values

# Crear una lista para almacenar las imágenes
imagenes = []

# Cargar las imágenes de las rutas de archivo
for ruta in rutas:
    imagen = tf.keras.preprocessing.image.load_img(ruta, target_size=(224, 224))
    imagen = tf.keras.preprocessing.image.img_to_array(imagen)
    imagenes.append(imagen)

# Convertir la lista de imágenes a un array numpy
imagenes = np.array(imagenes)

# Dividir los datos en conjuntos de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(imagenes, etiquetas, test_size=0.2, random_state=42)

# Normalizar los valores de píxeles entre 0 y 1
X_train = X_train / 255.0
X_val = X_val / 255.0

# Definir el modelo de red neuronal
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32)


# Obtener la precisión de entrenamiento y prueba del historial
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Crear el gráfico
epochs = range(1, len(train_acc) + 1)
plt.plot(epochs, train_acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Guardar el modelo entrenado
#model.save('modelo_huellas.h5')
