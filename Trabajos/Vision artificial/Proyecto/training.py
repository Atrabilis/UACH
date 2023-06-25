import os
import csv
import random
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout

# Limpia la consola
os.system("clear")

# Ruta de la carpeta que contiene las imágenes
carpeta_imagenes = 'DB2_B'

# Obtener la ruta completa del archivo CSV
ruta_csv = os.path.join(os.getcwd(), 'etiquetas.csv')

# Obtener la lista de archivos en la carpeta
archivos = os.listdir(carpeta_imagenes)

# Crear una lista para almacenar los nombres de archivo y etiquetas
datos = []

# Recorrer cada archivo en la carpeta
for archivo in archivos:
    # Obtener la ruta completa del archivo de imagen
    ruta_imagen = os.path.join(carpeta_imagenes, archivo)
    
    # Obtener la etiqueta basada en los primeros tres caracteres del nombre de archivo
    etiqueta = int(archivo[:3]) - 101
    
    # Agregar la ruta de imagen, nombre de archivo y la etiqueta a la lista
    datos.append([ruta_imagen, archivo, etiqueta])

# Guardar los nombres de archivo, etiquetas y rutas en un archivo CSV
with open(ruta_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Ruta', 'Archivo', 'Etiqueta'])
    writer.writerows(datos)

# Leer el archivo CSV y cargar los datos y etiquetas
datos = []
with open(ruta_csv, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        datos.append(row)

# Preparar los datos y etiquetas para el entrenamiento
X = []
y = []
for dato in datos:
    ruta_imagen = dato['Ruta']
    etiqueta = int(dato['Etiqueta'])
    
    # Cargar la imagen y convertirla a escala de grises
    imagen = Image.open(ruta_imagen).convert('L')
    
    # Cambiar el tamaño de la imagen a 328x364
    imagen = imagen.resize((328, 364))
    
    # Convertir la imagen a un array de numpy
    imagen = np.array(imagen)
    
    X.append(imagen)
    y.append(etiqueta)

# Convertir los datos y etiquetas a arrays de numpy
X = np.array(X)
y = np.array(y)

# Normalizar los valores de píxeles en el rango [0, 1]
X = X / 255.0

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de red neuronal
model = Sequential()
model.add(Flatten(input_shape=(364, 328)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# Compilar el modelo
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Definir el porcentaje objetivo de precisión
porcentaje_objetivo = 0.99

# Entrenar el modelo hasta alcanzar el porcentaje objetivo de precisión
epocas = 0
while epocas <= 1000:
    model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=2)
    epocas += 1
    _, accuracy = model.evaluate(X_test, y_test)
    if accuracy >= porcentaje_objetivo:
        break

# Imprimir el número de épocas necesarias para alcanzar el porcentaje objetivo
print('Épocas requeridas:', epocas)

# Hacer predicciones con el modelo
predicciones = model.predict(X_test)

# Obtener la clase predicha para cada ejemplo
clases_predichas = np.argmax(predicciones, axis=1)

# Imprimir las clases predichas
print(clases_predichas)
