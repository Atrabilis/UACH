from backpropagation import* #Incorpora clase con Backpropagation
import math # Sólo se utilizó para incorporar función seno y pi

# Parámetros de la red
# [R S1 S2] Agregar elementos según número de capas deseado
# Agregar una función de transferencia por cada capa de la red 1:sigmoidal, 2:tanh,  3:lineal.
red2 = Backprop([1, 2, 1], [2, 3])

red2.alfa = 0.4 # tasa de aprendizaje de la red
red2.num_epochs = 10000 # número máximo de épocas de entrenamiento
red2.target_error = 1E-12 # Error objetivo de entrenamiento

# Ejemplos de entrenamiento
P = np.array([[-2, -1.2, -0.4, 0.4, 1.2, 2]]) # Vector de entrada (Debe ser un arreglo 2D, por eso se utilizo doble paréntesis cuadrado)
T = np.empty((1,6)) # El arreglo T contiene las salidas esperadas para las correspondientes entradas P
for k in range(len(P[0,:])):
  T[0][k] = math.sin(P[0,k]*math.pi/4) # Llena el vector de salida para este ejemplo

#red2.W_init(2) # W_init se puede utilizar para inicializar todos los pesos con el valor deseado.
#red2.B_init(2) # B_init Se puede utilizar para inicializar todas las ganancias con el valor deseado.

# Valores iniciales de pesos y ganancias (para este ejemplo)
red2.W[1] = np.array([[-0.2], [0.5]])
red2.W[2] = np.array([[0.1, 0.3]])
red2.B[1] = np.array([[0.7], [-0.2]])
red2.B[2] = np.array([[0.8]])

red2.train(P, T) # entrena la red neuronal

print("La salida de la red para entrada 1.2 es ", red2.sim(1.2)) # Despliega en pantalla la salida de la red para una entrada 0.2
print(red2.W) # Despliega en pantalla los valores de los pesos de la red.
print(red2.B)  # Despliega en pantalla los valores de las ganancias de la red.