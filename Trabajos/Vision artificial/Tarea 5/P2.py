#Propósito del programa:
#Este programa implementa una red neuronal artificial con múltiples capas, utilizando la clase Neurona y las funciones de activación definidas en el archivo lib1.py.
#El propósito principal de la red neuronal es realizar el procesamiento de información en cada capa y generar las salidas correspondientes.

#Funcionamiento del programa:
#1. Se definen los datos de entrada y los pesos de las conexiones sinápticas para cada capa.
#2. Se crean las instancias de las neuronas en cada capa, especificando las entradas, pesos y función de activación correspondiente.
#3. Las neuronas en la capa de entrada reciben las entradas y los pesos definidos.
#4. Las neuronas en las capas ocultas y de salida realizan el procesamiento de información mediante la función de activación.
#5. Las salidas de las neuronas se utilizan como entradas para las siguientes capas, hasta llegar a la capa de salida.
#6. Se obtienen las salidas finales de la red neuronal.
#7. Las salidas de todas las neuronas se imprimen en pantalla.


import numpy as np  #Importa la librería numpy

from lib1 import *  #Importa la clase Neurona y las funciones de activación del archivo lib1.py

#Capa de entrada
neurona1 = Neurona(X1, pesos1, 0, escalon_unitario)  #Crea una neurona con entradas X1, pesos1 y función de activación escalon unitario
neurona2 = Neurona(X1, pesos2, 0, escalon_unitario)  #Crea una neurona con entradas X1, pesos2 y función de activación escalon_unitario
neurona3 = Neurona(X1, pesos3, 0, escalon_unitario)  #Crea una neurona con entradas X1, pesos3 y función de activación escalon_unitario

#Capa Escondida
neurona4 = Neurona(np.reshape([neurona1.salida(),
            neurona2.salida(),
            neurona3.salida()], (-1, 1)),
                   pesos4,
                   0,
                   escalon_unitario)  #Crea una neurona con entradas formadas por las salidas de las neuronas 1, 2 y 3,
                                      #pesos4 y función de activación escalon_unitario

neurona5 = Neurona(np.reshape([neurona1.salida(),
            neurona2.salida(),
            neurona3.salida()], (-1, 1)),
                   pesos5,
                   0,
                   escalon_unitario)  #Crea una neurona con entradas formadas por las salidas de las neuronas 1, 2 y 3,
                                      #pesos5 y función de activación escalon_unitario

#Capa Salida
neurona6 = Neurona(np.reshape([neurona4.salida(),
            neurona5.salida()], (-1, 1)),
                   pesos6,
                   0,
                   escalon_unitario)  #Crea una neurona con entradas formadas por las salidas de las neuronas 4 y 5,
                                      #pesos6 y función de activación escalon_unitario

#Imprime las salidas de todas las neuronas
print("Salida neurona 1: {}\n"
      "Salida neurona 2: {}\n"
      "Salida neurona 3: {}\n"
      "Salida neurona 4: {}\n"
      "Salida neurona 5: {}\n"
      "Salida neurona 6: {}\n".format(
          neurona1.salida(),
          neurona2.salida(),
          neurona3.salida(),
          neurona4.salida(),
          neurona5.salida(),
          neurona6.salida()
      ))