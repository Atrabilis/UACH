import matplotlib.pyplot as plt
def grafo_linea_desicion(w,b,inputs,labels):
    x0_min = -1
    x1_min = ( (-(w[0] * x0_min) - b[0]) 
            / w[1] )

    x0_max = 2
    x1_max = ( (-(w[0] * x0_max) - b[0]) 
            / w[1] )

    plt.plot([x0_min, x0_max], [x1_min, x1_max])
    plt.scatter(inputs[labels==0, 0], inputs[labels==0, 1], label='class 0', marker='o')
    plt.scatter(inputs[labels==1, 0], inputs[labels==1, 1], label='class 1', marker='s')

    plt.legend(loc='upper right')
    plt.show()