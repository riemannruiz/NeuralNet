import numpy as np

def f_tanh(x):
    # Funcion de activacion tangente hiperbolica
    y = np.tanh(x)
    dy = 1 - np.square(y)
    return y, dy