import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.applications.imagenet_utils import preprocess_input
import msvcrt

os.system("cls")

# Ruta de la carpeta que contiene las imágenes
carpeta_imagenes = 'Dataset'

# Cargar los datos del archivo CSV
datos = []
with open('labels.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Saltar la primera fila que contiene los encabezados
    for row in reader:
        datos.append(row)

# Separar los datos en imágenes (X) y etiquetas (y)
X = np.array([row[0] for row in datos])
y = np.array([int(row[2]) for row in datos])

# Dividir los datos en conjuntos de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Función para cargar y preprocesar una imagen con RLE
def cargar_y_preprocesar_imagen(ruta):
    imagen = Image.open(ruta).convert('L')  # Convertir la imagen a escala de grises
    imagen = imagen.resize((388, 374))  # Redimensionar la imagen a 388x374

    # Codificar la imagen con RLE
    pixels = np.array(imagen).ravel()  # Obtener los píxeles de la imagen como una secuencia 1D
    values = []
    lengths = []
    current_value = pixels[0]
    current_length = 1
    for pixel in pixels[1:]:
        if pixel == current_value:
            current_length += 1
        else:
            values.append(current_value)
            lengths.append(current_length)
            current_value = pixel
            current_length = 1
    values.append(current_value)
    lengths.append(current_length)

    # Normalizar los valores entre 0 y 1
    values = np.array(values) / 255.0

    return values, lengths

# Cargar y preprocesar los datos de entrenamiento y validación con RLE
X_train = [cargar_y_preprocesar_imagen(ruta) for ruta in X_train]
X_val = [cargar_y_preprocesar_imagen(ruta) for ruta in X_val]

# Obtener los valores y longitudes de los datos
X_train_values = [values for values, lengths in X_train]
X_train_lengths = [lengths for values, lengths in X_train]
X_val_values = [values for values, lengths in X_val]
X_val_lengths = [lengths for values, lengths in X_val]

# Crear el modelo de la red neuronal
model = Sequential()
model.add(Flatten(input_shape=(len(X_train_values[0]),)))  # Capa de entrada con la longitud de la secuencia RLE
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))  # Capa de salida con 10 neuronas y función de activación Softmax

# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Variables para controlar el bucle
epoch = 0
target_accuracy = 0.9  # Precisión objetivo en los datos de validación

# Listas para almacenar las precisiones en entrenamiento y validación
train_accuracies = []
val_accuracies = []

# Función para detectar si se presionó la tecla coma (",")
def key_pressed():
    return msvcrt.kbhit() and msvcrt.getch() == b','

# Bucle principal
while epoch <= 1000:
    # Entrenar el modelo durante una época
    history = model.fit(X_train_lengths, y_train, validation_data=(X_val_lengths, y_val), epochs=1, batch_size=32, verbose=1)

    # Obtener la precisión en los datos de entrenamiento y validación
    train_accuracy = history.history['accuracy'][0]  # Precisión en entrenamiento
    val_accuracy = history.history['val_accuracy'][0]  # Precisión en validación
    print(f'Precisión en entrenamiento en la época {epoch + 1}: {train_accuracy}')
    print(f'Precisión en validación en la época {epoch + 1}: {val_accuracy}')

    # Almacenar las precisiones en las listas correspondientes
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    # Verificar si se alcanzó la precisión objetivo o se presionó la tecla coma
    if val_accuracy >= target_accuracy or key_pressed():
        break

    epoch += 1

# Graficar el historial de precisión
plt.plot(train_accuracies)
plt.plot(val_accuracies)
plt.title('Historial de Precisión')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend(['Entrenamiento', 'Validación'], loc='upper left')
plt.show()

# Guardar el modelo
# model.save('model.h5')
