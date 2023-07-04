import cv2
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

os.system("cls")

# Cargar el archivo CSV con los datos procesados
csv_file = 'datos.csv'
df = pd.read_csv(csv_file)

# Aplicar aumentación de datos y agregar las nuevas imágenes y etiquetas al DataFrame
def apply_augmentation(df):
    augmented_data = []
    
    for index, row in df.iterrows():
        image_path = row["Contour Image"]
        label = row['Label']
        
        # Leer la imagen original
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Aplicar operaciones de aumentación
        augmented_images = []
        augmented_labels = []
        
        # Operación de rotación
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        augmented_images.append(rotated_image)
        augmented_labels.append(label)
        
        # Operación de espejo horizontal
        flipped_image = cv2.flip(image, 1)
        augmented_images.append(flipped_image)
        augmented_labels.append(label)
        
        # Operación de cambio de brillo
        brightness_image = np.clip(image + 50, 0, 255).astype(np.uint8)
        augmented_images.append(brightness_image)
        augmented_labels.append(label)

        contrast_image = np.clip(image * 1.5, 0, 255).astype(np.uint8)
        augmented_images.append(contrast_image)
        augmented_labels.append(label)
        
        scaled_image = cv2.resize(image, (0, 0), fx=1.2, fy=1.2)
        augmented_images.append(scaled_image)
        augmented_labels.append(label)

        # Operación de desplazamiento horizontal
        shifted_image = np.roll(image, 20, axis=1)
        augmented_images.append(shifted_image)
        augmented_labels.append(label)

        # Operación de desplazamiento vertical
        shifted_image = np.roll(image, 20, axis=0)
        augmented_images.append(shifted_image)
        augmented_labels.append(label)

        # Operación de cambio de perspectiva
        rows, cols = image.shape
        pts1 = np.float32([[10, 10], [cols - 10, 10], [10, rows - 10], [cols - 10, rows - 10]])
        pts2 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        perspective_image = cv2.warpPerspective(image, M, (cols, rows))
        augmented_images.append(perspective_image)
        augmented_labels.append(label)

        # Operación de ruido gaussiano
        mean = 0
        stddev = 10
        noise = np.random.normal(mean, stddev, image.shape)
        noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
        augmented_images.append(noisy_image)
        augmented_labels.append(label)

        # Operación de recorte aleatorio
        x = np.random.randint(0, cols - 20)
        y = np.random.randint(0, rows - 20)
        cropped_image = image[y:y+20, x:x+20]
        augmented_images.append(cropped_image)
        augmented_labels.append(label)

        # Agregar las imágenes y etiquetas aumentadas al conjunto de datos
        for augmented_image, augmented_label in zip(augmented_images, augmented_labels):
            augmented_data.append({'Ruta': image_path, 'Etiqueta': augmented_label, 'Imagen': augmented_image})
        
    # Crear un nuevo DataFrame con los datos aumentados
    augmented_df = pd.DataFrame(augmented_data)
    
    return augmented_df

augmented_df = apply_augmentation(df)

# Dividir los datos aumentados en conjuntos de entrenamiento, validación y prueba
train_df, test_df = train_test_split(augmented_df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Guardar los conjuntos de datos en archivos CSV separados
train_csv = 'train_data.csv'
val_csv = 'val_data.csv'
test_csv = 'test_data.csv'
train_df.to_csv(train_csv, index=False)
val_df.to_csv(val_csv, index=False)
test_df.to_csv(test_csv, index=False)

# Imprimir el número de muestras en cada conjunto de datos
print("Número de muestras antes del aumento de datos: ", len(df) )
print("Número de muestras en el conjunto de entrenamiento:", len(train_df))
print("Número de muestras en el conjunto de validación:", len(val_df))
print("Número de muestras en el conjunto de prueba:", len(test_df))
print("Número de muestras despues del aumento de datos: ", len(train_df)+len(val_df)+len(test_df))

