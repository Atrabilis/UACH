import numpy as np 
import matplotlib.pyplot as plt


class Backprop:

    def __init__(self, red_arquitectura, red_tfn):
        self.red_arquitectura = red_arquitectura
        self.red_tfn = red_tfn
        self.W = dict.fromkeys([i for i in range(1, len(red_arquitectura))]) # Crea diccionario para matrices de peso
        self.B = dict.fromkeys([i for i in range(1, len(red_arquitectura))]) # Crea diccionario para vectores de ganancia   
        self.W_init(-1) # Iniciaiza Pesos con valor -1
        self.B_init(-1) # Iniciaiza Ganancias con valor -1
        self.num_epochs = 1000
        self.alfa = 0.05
        self.target_error = 1E-5

    # FUNCIONES DE ACTIVACIÓN

    # Función Sigmoidal
    def sigm(self, x):
        return 1/(1 + np.exp(-x))

    # Primera derivada Función Sigmoidal
    def sigm_der(self, x):
        return self.sigm(x)*(1 - self.sigm(x))

    # Función Tangente Hiperbólica
    def tanh(self, x):
        return np.tanh(x)

    # Primera derivada Función Tangente Hiperbólica
    def tanh_der(self, x):
        return 1 - self.tanh(x)**2

    # Función Lineal
    def lineal(self, x):
        return x

    # Primera derivada Función Lineal
    def lineal_der(self, x):
        return 1

    # Multiplica matrices y escalares
    def matMult(self, A, B):
        if np.size(A) > 1 and np.size(B) > 1:
            return np.matmul(A, B)
        else:
            return np.multiply(A, B)

    # Aplica función de transferencia
    def TFn(self, x, index):
        if index == 1: return self.sigm(x)
        elif index == 2: return self.tanh(x) 
        else: return self.lineal(x)

    # Aplica primera derivada de función de transferencia
    def TFnDer(self, x, index):
        if index == 1: return self.sigm_der(x)
        elif index == 2: return self.tanh_der(x) 
        else: return self.lineal_der(x)

    # Inicializa Pesos con valor n
    def W_init(self, n):  
        for k in range(len(self.red_arquitectura) - 1):
            w_array = np.ones((self.red_arquitectura[k + 1], self.red_arquitectura[k])) # Inicializa matrices de peso con valor 1
            self.W[k + 1] = n*w_array

    # Inicializa Ganancias con valor n
    def B_init(self, n):  
        for k in range(len(self.red_arquitectura) - 1):
            b_array = np.ones([self.red_arquitectura[k + 1], 1]) # Inicializa vectores de ganancia con valor 1
            self.B[k + 1] = n*b_array

    # Cálculo de salida de red (Feed-Forward)
    def sim(self, p):
        for k in range(len(self.red_tfn)):
            n = self.matMult(self.W[k + 1], p)  + self.B[k + 1]
            p = self.TFn(n, self.red_tfn[k])
        return p

    # Cálculo de argumentos de funciones de activación
    def CalcN(self, N, p):
        for k in range(len(self.red_tfn)):
            N[k + 1] = self.matMult(self.W[k + 1], p) + self.B[k + 1]
            p = self.TFn(N[k + 1], self.red_tfn[k])

    # Entrenamiento de la red
    def train(self, P, T):

        N = dict.fromkeys([i for i in range(1, len(self.red_arquitectura))]) # Crea diccionario para N (argumentos de función de activación en cada neurona)
        So = dict.fromkeys([i for i in range(1, len(self.red_arquitectura) - 1)]) # Crea diccionario para Sensibilidades de Capas Ocultas
        index_capa_salida = len(self.red_arquitectura) - 1

        # Datos para gráfico de entrenamiento
        G = np.empty(self.num_epochs)
    
        min_error = 1E80 # Error mínimo actual
        index_min_error = 0 # Índice de error mínimo actual

        num_ejem_entren = len(T[0][:])
        epoch = 0
        
        while epoch < self.num_epochs:
        
            for index in range(num_ejem_entren):
       
                self.CalcN(N, P[:,[index]])

                # Cálculo de salida (Feed-Forward)
                a = self.sim(P[:,[index]])
                
                # Cálculo de sensibilidad de capa de salida
                Ss = (T[:,[index]] - a) * self.TFnDer(N[index_capa_salida - 1], self.red_tfn[index_capa_salida - 1])
           
                # Cálculo de sensibilidad de capa oculta
                for layer in range(index_capa_salida - 1, 0, -1):
                    So[layer] = np.empty((self.red_arquitectura[layer],1))
                    for j in range(0, self.red_arquitectura[layer]):
                        suma = 0
                        for k in range(self.red_arquitectura[layer + 1]):
                            if layer < index_capa_salida - 1: suma = suma + So[layer + 1][k]*self.W[layer + 1][k][j]
                            else: 
                                suma = suma + Ss[k]*self.W[layer + 1][k][j]
                        So[layer][j] = suma*self.TFnDer(N[layer][j], self.red_tfn[layer - 1])
                
                # Actualización de Pesos y Ganancias
                for layer in range(index_capa_salida, 0, -1):
                        
                    if layer < index_capa_salida and layer > 1:
                        self.W[layer] = self.W[layer] + self.alfa*So[layer]*np.transpose(self.TFn(N[layer - 1], self.red_tfn[layer - 2]))      
                    elif layer == 1:
                        self.W[1] = self.W[1] + self.alfa*So[layer]*np.transpose(P[:,index])  
                    else:
                        self.W[layer] = self.W[layer] + self.alfa*Ss*a
                                
                    if layer < index_capa_salida:
                        self.B[layer] = self.B[layer] + self.alfa*So[layer]
                    else:
                        self.B[layer] = self.B[layer] + self.alfa*Ss
                    
            # Cálculo de error
            e2 = 0
            for index in range(num_ejem_entren):
                ep2 = 0
                # Cálculo de salida (Feed-Forward)
                a = self.sim(P[:,[index]])
                for k in range(self.red_arquitectura[index_capa_salida]):
                    error = (T[k,index] - a[k])**2
                    ep2 = ep2 + error[0]
                e2 = e2 + ep2/2

            G[epoch] = e2
            epoch = epoch + 1

            if e2 < min_error: 
                min_error = e2
                index_min_error = epoch
            if e2 < self.target_error: break

        # Gráfico de entrenamiento
        plt.figure(num = "Training results")
        plt.plot([x for x in range(1, epoch + 1)], G[0:epoch], color = '#0040ff')
        plt.title("Error final = " +  str("{:.4e}".format(e2)))
        plt.xlabel("Epoch")
        plt.ylabel("Error medio cuadrático (mse)")
        plt.yscale('log')
        plt.grid()
        plt.plot(index_min_error, min_error, 'o', ms = 7, color = 'g')
        print("Error mínimo: " + str("{:.4e}".format(min_error)) + " en época ", index_min_error)
        plt.show()