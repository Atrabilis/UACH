import cv2
import csv
import numpy as np
import os
from sklearn.model_selection import train_test_split

os.system("cls")

def extract_minutiae(image_path, save_images=False):
    # Cargar la imagen en escala de grises
    image = cv2.imread(image_path, 0)

    # Aplicar ecualización del histograma
    image_equalized = cv2.equalizeHist(image)

    # Aplicar un filtro de suavizado para reducir el ruido
    image_blurred = cv2.GaussianBlur(image_equalized, (5, 5), 0)

    # Aplicar umbralización adaptativa para binarizar la imagen
    _, image_thresholded = cv2.threshold(image_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Aplicar operaciones morfológicas para mejorar la imagen binaria
    kernel = np.ones((3, 3), np.uint8)
    image_eroded = cv2.erode(image_thresholded, kernel, iterations=2)
    image_dilated = cv2.dilate(image_eroded, kernel, iterations=1)

    # Encontrar los contornos de las crestas de la huella
    contours, _ = cv2.findContours(image_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujar los contornos en la imagen original
    image_with_contours = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)

    # Extraer las características de las crestas (minutias)
    minutiae = []
    for contour in contours:
        hull = cv2.convexHull(contour, clockwise=False, returnPoints=True)
        for i in range(1, len(hull) - 1, 5):
            start = tuple(hull[i-1][0])
            end = tuple(hull[i+1][0])
            minutiae.append((start, end))

    if save_images:
        # Crear la carpeta "contornos" si no existe
        if not os.path.exists('contornos'):
            os.makedirs('contornos')

        # Guardar la imagen con los contornos en la carpeta "contornos"
        contours_image_path = os.path.join('contornos', os.path.basename(image_path))
        cv2.imwrite(contours_image_path, image_with_contours)

        # Crear la carpeta "minutias" si no existe
        if not os.path.exists('minutias'):
            os.makedirs('minutias')

        # Dibujar las minutias sobre la imagen original
        image_with_minutiae = image.copy()
        for (start, end) in minutiae:
            cv2.circle(image_with_minutiae, start, 5, (0, 0, 255), 2)
            cv2.line(image_with_minutiae, start, end, (0, 0, 255), 2)

        # Guardar la imagen con las minutias en la carpeta "minutias"
        minutiae_image_path = os.path.join('minutias', os.path.basename(image_path))
        cv2.imwrite(minutiae_image_path, image_with_minutiae)

    return image_with_contours, contours, minutiae, contours_image_path, minutiae_image_path

# Ruta de la carpeta con las imágenes
dataset_folder = 'dataset'

# Nombre del archivo CSV
csv_filename = 'datos.csv'

# Listas para almacenar los datos
filenames = []
contours_image_paths = []
minutiae_image_paths = []
etiquetas = []

# Extraer las minutias y mostrar los contornos y minutias de cada imagen en la carpeta
for filename in os.listdir(dataset_folder):
    if filename.endswith('.tif'):
        image_path = os.path.join(dataset_folder, filename)
        image_contours, contours, minutiae, contours_image_path, minutiae_image_path = extract_minutiae(image_path, save_images=True)

        # Determinar la etiqueta de la imagen basándose en el nombre del archivo
        etiqueta = int(filename.split('_')[0])-101

        # Agregar los datos a las listas
        filenames.append(filename)
        contours_image_paths.append(contours_image_path)
        minutiae_image_paths.append(minutiae_image_path)
        etiquetas.append(etiqueta)

        # Mostrar el número de minutias extraídas y las coordenadas de las minutias
        print("Minutias extraídas de", filename)
        print("Número de minutias extraídas:", len(minutiae))
        for i, (start, end) in enumerate(minutiae):
            print("Minutia", i+1, ": Start:", start, "End:", end)

        print("-------------------------------------")

# Dividir los datos en conjuntos de entrenamiento, validación y prueba
train_val_filenames, test_filenames, train_val_contours_image_paths, test_contours_image_paths, train_val_minutiae_image_paths, test_minutiae_image_paths, train_val_etiquetas, test_etiquetas = train_test_split(filenames, contours_image_paths, minutiae_image_paths, etiquetas, test_size=0.2, random_state=42)
train_filenames, val_filenames, train_contours_image_paths, val_contours_image_paths, train_minutiae_image_paths, val_minutiae_image_paths, train_etiquetas, val_etiquetas = train_test_split(train_val_filenames, train_val_contours_image_paths, train_val_minutiae_image_paths, train_val_etiquetas, test_size=0.25, random_state=42)

# Abrir el archivo CSV en modo escritura
with open(csv_filename, 'w', newline='') as file:
    writer = csv.writer(file)

    # Escribir la cabecera del archivo CSV
    writer.writerow(['Imagen', 'Ruta de Contornos', 'Ruta de Minutias', 'Etiqueta'])

    # Escribir los datos de entrenamiento en el archivo CSV
    for i in range(len(train_filenames)):
        writer.writerow([train_filenames[i], train_contours_image_paths[i], train_minutiae_image_paths[i], train_etiquetas[i]])

    # Escribir los datos de validación en el archivo CSV
    for i in range(len(val_filenames)):
        writer.writerow([val_filenames[i], val_contours_image_paths[i], val_minutiae_image_paths[i], val_etiquetas[i]])

    # Escribir los datos de prueba en el archivo CSV
    for i in range(len(test_filenames)):
        writer.writerow([test_filenames[i], test_contours_image_paths[i], test_minutiae_image_paths[i], test_etiquetas[i]])

    # Mostrar la información de la separación de datos
    print("Número de datos de entrenamiento:", len(train_filenames))
    print("Número de datos de validación:", len(val_filenames))
    print("Número de datos de prueba:", len(test_filenames))
