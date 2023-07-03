import pandas as pd
from sklearn.model_selection import train_test_split

# Leer el archivo CSV con los datos
df = pd.read_csv('datos.csv')

# Dividir los datos en conjunto de entrenamiento, validación y prueba
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Guardar los conjuntos de datos en archivos CSV separados
train_df.to_csv('train.csv', index=False)
val_df.to_csv('validation.csv', index=False)
test_df.to_csv('test.csv', index=False)