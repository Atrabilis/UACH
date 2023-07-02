import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np
import matplotlib.pyplot as plt
import keyboard

# Limpiar la consola
os.system('cls' if os.name == 'nt' else 'clear')

# Rutas de los archivos CSV
train_file = 'train_data.csv'
val_file = 'val_data.csv'
test_file = 'test_data.csv'

# Leer los archivos CSV en DataFrames
train_data = pd.read_csv(train_file)
val_data = pd.read_csv(val_file)
test_data = pd.read_csv(test_file)

# Tamaño deseado de las imágenes
target_size = (50, 50)

# Función para cargar y redimensionar las imágenes
def load_and_preprocess_image(image_path):
    image = load_img(image_path, target_size=target_size)  # Redimensionar la imagen al tamaño deseado
    image_array = img_to_array(image) / 255.0  # Normalizar los valores de píxel entre 0 y 1
    return image_array

# Obtener las rutas de las imágenes de entrenamiento, las etiquetas correspondientes y convertir las etiquetas en one-hot encoding
train_images = [load_and_preprocess_image(image_path) for image_path in train_data['Ruta']]
train_images = np.array(train_images)
train_labels = tf.keras.utils.to_categorical(train_data['Etiqueta'])

# Obtener las rutas de las imágenes de validación, las etiquetas correspondientes y convertir las etiquetas en one-hot encoding
val_images = [load_and_preprocess_image(image_path) for image_path in val_data['Ruta']]
val_images = np.array(val_images)
val_labels = tf.keras.utils.to_categorical(val_data['Etiqueta'])

# Obtener las rutas de las imágenes de prueba y las etiquetas correspondientes
test_images = [load_and_preprocess_image(image_path) for image_path in test_data['Ruta']]
test_images = np.array(test_images)
test_labels = tf.keras.utils.to_categorical(test_data['Etiqueta'])

# Crear el modelo de la red neuronal
model = Sequential()
model.add(Flatten(input_shape=train_images.shape[1:]))  # Aplanar la imagen en un vector
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))  # Capa oculta con 64 neuronas
model.add(Dense(40, activation='softmax'))  # Capa de salida con activación softmax

# Compilar el modelo
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Definir una función para detener el entrenamiento
def stop_training():
    model.stop_training = True

# Asignar la función stop_training a la tecla F10
keyboard.on_press_key("f10", lambda _: stop_training())

# Entrenar el modelo
history = model.fit(train_images, train_labels, epochs=500, batch_size=64, validation_data=(val_images, val_labels),
                    callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: stop_training() if model.stop_training else None)])

# Evaluar el modelo en el conjunto de prueba
#loss, accuracy = model.evaluate(test_images, test_labels)
#print('Test loss:', loss)
#print('Test accuracy:', accuracy)

# Mostrar el gráfico del historial de precisión
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
