import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import csv

# Limpiar la pantalla
os.system("cls")

# Funciones de procesamiento de imágenes
def fft_enhance(image, f):
    I = 255 - image.astype(np.float64)
    w, h = I.shape
    w1 = w // 32 * 32
    h1 = h // 32 * 32
    inner = np.zeros((w1, h1))
    
    for i in range(0, w1, 32):
        for j in range(0, h1, 32):
            a = i + 31
            b = j + 31
            F = np.fft.fft2(I[i:a, j:b])
            factor = np.abs(F) ** f
            block = np.abs(np.fft.ifft2(F * factor))
            larv = np.max(block)
            
            if larv == 0:
                larv = 1
            
            block = block / larv
            inner[i:a, j:b] = block
    
    final = inner * 255
    final = cv2.equalizeHist(final.astype(np.uint8))
    
    return final

def adaptiveBinarization(image, block_size):
    binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, 0)
    return binary_image


# Cargar el archivo CSV
csv_file = 'data.csv'
df = pd.read_csv(csv_file)

# Procesar las imágenes y guardarlas en una carpeta
processed_folder = 'Procesadas'
os.makedirs(processed_folder, exist_ok=True)

# Crear una lista para los datos del nuevo CSV
new_csv_data = []

for index, row in df.iterrows():
    image_path = row['Ruta']
    label = row['Etiqueta']

    # Leer la imagen
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Aplicar fft_enhance
    processed_image = fft_enhance(image, 0.45)  # Utilizar el valor 0.45 como segundo parámetro

    # Aplicar adaptiveThres
    processed_image = adaptiveBinarization(processed_image, 17)

    # Obtener el nombre del archivo sin extensión
    file_name = os.path.splitext(os.path.basename(image_path))[0] + '.jpg'

    # Guardar la imagen procesada
    processed_image_path = os.path.join(processed_folder, file_name.replace('.tif', '.jpg'))
    cv2.imwrite(processed_image_path, processed_image)
    print("imagen {} guardada".format(file_name))

    # Agregar los datos al nuevo CSV
    new_csv_data.append([processed_image_path, file_name, label])

# Guardar los datos en el nuevo CSV
new_csv_file = 'processed_data.csv'
header = ['Ruta', 'Archivo', 'Etiqueta']

with open(new_csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(new_csv_data)

# Mostrar el DataFrame con las rutas de las imágenes procesadas
print("Imagenes procesadas con éxito")


