import numpy as np

def f_lineal(x):
    # Funcion de activacion lineal
    y = x
    dy = np.ones_like(x)
    return y, dy