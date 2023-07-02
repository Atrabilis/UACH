import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Cargar el archivo CSV con los datos procesados
csv_file = 'data.csv'
df = pd.read_csv(csv_file)

# Aplicar aumentación de datos y agregar las nuevas imágenes y etiquetas al DataFrame
def apply_augmentation(df):
    augmented_data = []
    
    for index, row in df.iterrows():
        image_path = row['Ruta']
        label = row['Etiqueta']
        
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
        
        # Agregar las imágenes y etiquetas aumentadas al conjunto de datos
        for augmented_image, augmented_label in zip(augmented_images, augmented_labels):
            augmented_data.append({'Ruta': image_path, 'Etiqueta': augmented_label, 'Imagen': augmented_image})
        
    # Crear un nuevo DataFrame con los datos aumentados
    augmented_df = pd.DataFrame(augmented_data)
    
    return augmented_df

augmented_df =  df        #############apply_augmentation(df)

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
print("Número de muestras en el conjunto de entrenamiento:", len(train_df))
print("Número de muestras en el conjunto de validación:", len(val_df))
print("Número de muestras en el conjunto de prueba:", len(test_df))
