import numpy as np

def neural_layer(X, W, b, f_activ):

    V = np.dot(X, W.T) + np.dot(np.ones((X.shape[0], 1)), b.T)   #Calculo del umbral de activacion
    Y, phi_ = f_activ(V) # Calculo de la salida de capa neuronal
    return Y, V,phi_
