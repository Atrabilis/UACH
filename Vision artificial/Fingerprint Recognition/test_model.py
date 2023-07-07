import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

# Cargar el modelo guardado
model = load_model('modelo_entrenado.h5')

# Función para cargar y preprocesar la imagen
def load_and_preprocess_image(image_path):
    image = load_img(image_path, target_size=(50, 50), color_mode='grayscale')
    image_array = img_to_array(image) / 255.0
    return image_array

# Cargar las etiquetas de prueba
test_data_df = pd.read_csv('datos.csv')
image_files = test_data_df['Contour Image'].values
true_labels = test_data_df['Label'].values

predicted_classes = []
predicted_probs = []

# Iterar sobre cada imagen y realizar la predicción
for image_file in image_files:
    # Cargar y preprocesar la imagen
    image = load_and_preprocess_image(image_file)
    
    # Realizar la predicción
    predictions = model.predict(np.expand_dims(image, axis=0))
    
    # Obtener la clase predicha
    predicted_class = np.argmax(predictions)
    
    # Obtener la probabilidad asociada a la clase predicha
    confidence = predictions[0][predicted_class]
    
    # Agregar a las listas
    predicted_classes.append(predicted_class)
    predicted_probs.append(confidence)
    
    # Imprimir los resultados
    print('Imagen:', image_file)
    print('Clase predicha:', predicted_class)
    print('Probabilidad:', confidence)
    print('---')

# Calcular e imprimir las métricas
print('Accuracy:', accuracy_score(true_labels, predicted_classes))
print('Confusion Matrix:\n', confusion_matrix(true_labels, predicted_classes))
print('Classification Report:\n', classification_report(true_labels, predicted_classes))

# Calcular la matriz de confusión
cm = confusion_matrix(true_labels, predicted_classes)

# Crear una figura y un conjunto de ejes
fig, ax = plt.subplots(figsize=(10, 10))

# Crear un objeto de visualización de la matriz de confusión
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

# Plot the confusion matrix
disp.plot(cmap='Blues', ax=ax)

# Agregar título y etiquetas
ax.set_title('Matriz de Confusión')
plt.xlabel('Clase Predicha')
plt.ylabel('Clase Verdadera')

# Mostrar la figura
plt.show()

# Crear una figura y un conjunto de ejes
fig, ax = plt.subplots(figsize=(10, 10))

# Crear un histograma de las probabilidades de predicción
sns.histplot(predicted_probs, bins=10, kde=False, ax=ax)

# Agregar título y etiquetas
ax.set_title('Distribución de las Probabilidades de Predicción')
plt.xlabel('Probabilidad de Predicción')
plt.ylabel('Conteo')

# Mostrar la figura
plt.show()
