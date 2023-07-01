import os
import csv
import random

os.system("cls")

# Ruta de la carpeta que contiene las imágenes
carpeta_imagenes = 'Dataset'

# Obtener la lista de archivos en la carpeta
archivos = os.listdir(carpeta_imagenes)

# Crear una lista para almacenar los nombres de archivo, rutas y etiquetas
datos = []

# Recorrer cada archivo en la carpeta
for archivo in archivos:
    # Obtener la ruta completa del archivo
    ruta = os.path.join(carpeta_imagenes, archivo)
    
    # Obtener los primeros tres caracteres del nombre de archivo y convertirlo a entero
    etiqueta = int(archivo[:3])
    
    # Agregar la ruta de archivo, nombre de archivo y etiqueta a la lista
    datos.append([ruta, archivo, etiqueta -101])

# Guardar los nombres de archivo, rutas y etiquetas en un archivo CSV
with open('Data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Ruta', 'Archivo', 'Etiqueta'])
    writer.writerows(datos)
