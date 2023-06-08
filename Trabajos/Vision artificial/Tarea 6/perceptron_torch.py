import numpy as np
import matplotlib.pyplot as plt
import torch

# Implementación del Perceptrón con PyTorch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Permite realizar procesamiento en GPU (Graphics Processing Unit)

class Perceptron():
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = torch.zeros(num_features, 1, 
                                   dtype=torch.float32, device=device)
        self.bias = torch.zeros(1, dtype=torch.float32, device=device)
        
        self.ones = torch.ones(1, device=device)
        self.zeros = torch.zeros(1, device=device)

    def forward(self, x):
        linear = torch.mm(x, self.weights) + self.bias
        predictions = torch.where(linear > 0., self.ones, self.zeros)
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
        accuracy = torch.sum(predictions == y).float() / y.shape[0]
        return accuracy

# Entrenamiento del Perceptrón

ppn = Perceptron(num_features=2) # num_features es el número de componentes del vector de entrada

X_train = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32, device=device) # Ingresar entradas como vector fila. Un tensor en PyTorch es básicamente lo mismo que un array en NumPy.
y_train = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=device) # Ingresar salidas deseadas para cada entrada

ppn.train(X_train, y_train, epochs=5) # epochs son el número de iteraciones del algoritmo de entrenamiento (en cada iteración se aplica la regla de aprendizaje a todos los ejemplos de entrenamiento)

print('Parámetros del modelo:\n\n')
print('  Pesos: %s\n' % ppn.weights) # Despliega los pesos del perceptrón
print('  Bias: %s\n' % ppn.bias) # Despliega el bias del perceptrón

# Evaluación del modelo

train_acc = ppn.evaluate(X_train, y_train)
print('Precisión del entrenamiento: %.2f%%' % (train_acc*100)) # Despliega la precisión del entrenamiento

test_data = torch.tensor([[0.99, 1.001]], dtype=torch.float32, device=device)
sim = ppn.forward(test_data)
print('Salida particular del perceptrón: ', sim)

# Gráfico 2D de recta de decisión

w, b = ppn.weights, ppn.bias

x0_min = -0.5
x1_min = ( (-(w[0] * x0_min) - b[0]) 
          / w[1] )

x0_max = 1.5
x1_max = ( (-(w[0] * x0_max) - b[0]) 
          / w[1] )

# Traspasa procesamiento a CPU y transforma tensor PyTorch en array NumPy.
x1_min2 = x1_min.cpu()
x1_min2 = np.array(x1_min2)
x1_max2 = x1_max.cpu()
x1_max2 = np.array(x1_max2)
X_train2 = X_train.cpu()
X_train2 = np.array(X_train2)
y_train2 = y_train.cpu()
y_train2 = np.array(y_train2)

plt.plot([x0_min, x0_max], [np.array(x1_min2), np.array(x1_max2)])
plt.scatter(X_train2[y_train2==0, 0], X_train2[y_train2==0, 1], label='class 0', marker='o')
plt.scatter(X_train2[y_train2==1, 0], X_train2[y_train2==1, 1], label='class 1', marker='s')

plt.legend(loc='upper right')
plt.show()