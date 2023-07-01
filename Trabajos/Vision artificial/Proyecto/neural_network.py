import tensorflow as tf
import pandas as pd
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import keyboard


# Limpiar la consola
os.system("cls")

# Función para procesar las imágenes
def procesar_imagen(ruta_imagen):
    # Cargar la imagen en escala de grises
    imagen = cv2.imread(ruta_imagen, 0)

    # Aplicar ecualización de histograma
    ecualizada = cv2.equalizeHist(imagen)

    # Aplicar binarización local adaptativa
    ventana = 31
    constante = 5
    binarizada = cv2.adaptiveThreshold(ecualizada, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, ventana, constante)

    # Aplicar el filtro de Sobel
    sobel_x = cv2.Sobel(binarizada, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(binarizada, cv2.CV_64F, 0, 1, ksize=3)

    # Calcular el gradiente de orientación
    gradiente_orientacion = np.arctan2(sobel_y, sobel_x)

    return gradiente_orientacion


# Cargar los datos desde el archivo CSV
data = pd.read_csv('data.csv')

# Separar características (ruta y nombre del archivo) y etiquetas
features = data[['Ruta', 'Archivo']]
labels = data['Etiqueta']

# Codificar las etiquetas como números
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Obtener el número de etiquetas
num_labels = len(label_encoder.classes_)

# Procesar las imágenes y ajustar su tamaño
processed_features = []
max_image_size = (100, 100)  # Tamaño máximo deseado para las imágenes
for ruta_imagen in features['Ruta']:
    processed_image = procesar_imagen(ruta_imagen)
    resized_image = cv2.resize(processed_image, max_image_size)
    processed_features.append(resized_image)

# Convertir a un formato numpy array
features_processed = np.array(processed_features)

# Dividir los datos en conjuntos de entrenamiento y prueba
train_features, test_features, train_labels, test_labels = train_test_split(features_processed, labels_encoded, test_size=0.2)

# Crear la red neuronal
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=features_processed.shape[1:]),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Capa de Dropout con una tasa de 0.5
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(num_labels, activation='softmax')  # Capa de salida con el número de etiquetas
])

# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Definir un callback para detener el entrenamiento cuando se alcance un 90% de precisión en validación o al presionar la tecla coma (,)
class StopTrainingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs['val_accuracy'] >= 0.9 or keyboard.is_pressed(','):
            self.model.stop_training = True

# Entrenar el modelo con el callback definido
history = model.fit(train_features, train_labels, epochs=5000, batch_size=128, validation_data=(test_features, test_labels), callbacks=[StopTrainingCallback()])

# Obtener el historial de precisión en entrenamiento y validación
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Graficar el historial de precisión
epochs = range(1, len(train_accuracy) + 1)
plt.plot(epochs, train_accuracy, 'b', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
