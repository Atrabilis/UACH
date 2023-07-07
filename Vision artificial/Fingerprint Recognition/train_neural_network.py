from sklearn.model_selection import train_test_split
from keras.models import Sequential
from tensorflow.keras.utils import plot_model
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
import csv
import cv2
import os
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Specify the Graphviz executable path
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin' 

os.system("cls")

# Cargar los archivos CSV con los datos de entrenamiento, prueba y validación
train_csv = 'train_data.csv'
val_csv = 'val_data.csv'
test_csv = 'test_data.csv'

train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)
test_df = pd.read_csv(test_csv)

# Preprocesar las imágenes y convertirlas en arreglos NumPy
def preprocess_images(image_paths):
    images = []
    for path in image_paths:
        image = cv2.imread(str(path), 0)
        image = cv2.resize(image, (50, 50))  # Redimensionar a 50x50 píxeles
        image = image.astype('float32') / 255.0  # Normalizar los valores de los píxeles en el rango [0, 1]
        images.append(image)
    images = np.array(images)
    return images

X_train_images = preprocess_images(train_df['Ruta'].values)
X_val_images = preprocess_images(val_df['Ruta'].values)
X_test_images = preprocess_images(test_df['Ruta'].values)

# Convertir las etiquetas a valores numéricos y aplicar one-hot encoding
labels, unique_labels = np.unique(train_df['Etiqueta'].values, return_inverse=True)
y_train_encoded = to_categorical(unique_labels)
y_val_encoded = to_categorical(np.unique(val_df['Etiqueta'].values, return_inverse=True)[1])
y_test_encoded = to_categorical(np.unique(test_df['Etiqueta'].values, return_inverse=True)[1])

# Crear el modelo de la red neuronal
model = Sequential()
model.add(Flatten(input_shape=(50, 50, 1)))  # Aplanar las características (50x50 imágenes en escala de grises)
model.add(Dense(512, activation='relu'))  # Capa oculta con 512 unidades y función de activación ReLU
model.add(Dense(40, activation='softmax'))  # Capa de salida con 40 unidades y función de activación softmax

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
epocas = 200
history = model.fit(X_train_images, y_train_encoded, epochs=epocas, batch_size=512, validation_data=(X_val_images, y_val_encoded), verbose=1)

# Evaluar el modelo en el conjunto de prueba
loss, accuracy = model.evaluate(X_test_images, y_test_encoded)

# Obtener historial de precisión y pérdida
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
test_accuracy = accuracy
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Crear gráfico
plt.figure(figsize=(10, 6))  # Ajustar tamaño del gráfico
plt.plot(train_accuracy, label='Precisión de entrenamiento')
plt.plot(val_accuracy, label='Precisión de validación')
plt.axhline(test_accuracy, color='r', linestyle='--', label='Precisión de prueba')
plt.plot(train_loss, label='Pérdida de entrenamiento')
plt.plot(val_loss, label='Pérdida de validación')
plt.xlabel('Épocas', fontsize=25)
plt.ylabel('Valor', fontsize=25)
plt.title('Métricas de Entrenamiento, Validación y Prueba', fontsize=25)
plt.legend(fontsize=20)  # Mostrar leyenda
plt.grid(True)  # Agregar cuadrícula al gráfico
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()


# Listado exhaustivo de métricas
print("Train Accuracy:", train_accuracy[-1])
print("Validation Accuracy:", train_accuracy[-1])
print("Train Loss:", train_loss[-1])
print("Validation Loss:", val_loss[-1])
print("Test Accuracy:", accuracy)
print("Test Loss:", loss)


# Cargar el modelo
model1 = tf.keras.models.load_model('modelo_entrenado.h5')

# Generar la representación gráfica del modelo
plot_model(model, to_file='model.png', show_shapes=True, show_dtype=False, show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96, layer_range=None, show_layer_activations=True)


# Guardar el modelo entrenado
model.save('modelo_entrenado.h5')