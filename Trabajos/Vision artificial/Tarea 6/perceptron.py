import numpy as np
import matplotlib.pyplot as plt

# Implementación del Perceptrón con NumPy

class Perceptron():
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = np.zeros((num_features, 1), dtype=float)
        self.bias = np.zeros(1, dtype=float)

    def forward(self, x):
        linear = np.dot(x, self.weights) + self.bias 
        predictions = np.where(linear > 0., 1, 0)
        return predictions
        
    def backward(self, x, y):  
        predictions = self.forward(x)
        errors = y - predictions
        return errors
        
    def train(self, x, y, epochs):
        for e in range(epochs):
            
            for i in range(y.shape[0]):
                errors = self.backward(x[i].reshape(1, self.num_features), y[i]).reshape(-1)
                self.weights += (errors * x[i]).reshape(self.num_features, 1)
                self.bias += errors
                
    def evaluate(self, x, y):
        predictions = self.forward(x).reshape(-1)
        accuracy = np.sum(predictions == y) / y.shape[0]
        return accuracy


# Entrenamiento del Perceptrón

ppn = Perceptron(num_features=2) # num_features es el número de componentes del vector de entrada

X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # Ingresar entradas como vector fila
y_train = np.array([0, 0, 0, 1]) # Ingresar salidas deseadas para cada entrada

ppn.train(X_train, y_train, epochs=5) # epochs son el número de iteraciones del algoritmo de entrenamiento (en cada iteración se aplica la regla de aprendizaje a todos los ejemplos de entrenamiento)

print('Parámetros del modelo:\n\n')
print('  Pesos: %s\n' % ppn.weights) # Despliega los pesos del perceptrón
print('  Bias: %s\n' % ppn.bias) # Despliega el bias del perceptrón

# Evaluación del modelo

train_acc = ppn.evaluate(X_train, y_train)
print('Precisión del entrenamiento: %.2f%%' % (train_acc*100)) # Despliega la precisión del entrenamiento

sim = ppn.forward([0.99, 1.001])
print('Salida particular del perceptrón: ', sim)


# Gráfico 2D de recta de decisión


w, b = ppn.weights, ppn.bias

x0_min = -1
x1_min = ( (-(w[0] * x0_min) - b[0]) 
          / w[1] )

x0_max = 2
x1_max = ( (-(w[0] * x0_max) - b[0]) 
          / w[1] )

plt.plot([x0_min, x0_max], [x1_min, x1_max])
plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], label='class 0', marker='o')
plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], label='class 1', marker='s')

plt.legend(loc='upper right')
plt.show()