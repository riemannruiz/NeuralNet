import numpy as np
def f_sigmoid(x):
    # Funcion de activacion sigmoidal
    y = 1.0 / (1.0 + np.exp(-x))
    dy = y * (1 - y)
    return y, dy