import numpy as np


def cost_fun_rmse_reg_(Y, Yhat, params):
    n_sample = Y.shape[0] # numero de muestras en el conjunto de datos
    lambda_ = params.lambda_reg # coeficiente de regularizacion

    Wreg = np.zeros((params.nw_total_layer, 1))

    idx_w = 0

    for k in range(len(params.W)):
        Wtemp = params.W[k].ravel()  # Acomodar los pesos en una columna
        fw = len(Wtemp)  # Contar el número de pesos
        Wreg[idx_w:idx_w + fw, 0] = Wtemp  # Creación de vector de pesos para regularización
        idx_w += fw
    #Error de estimacion de la red

    E = Y - Yhat
    Ef = E.ravel()
    #funcion de costo
    J = (Ef.T @ Ef) / (2 * n_sample * params.n_out) + (lambda_ / 2) * (Wreg.T @ Wreg)

    # Gradiente de funcion de costo con la salida de la red
    dJ_dY = -E / (n_sample * params.n_out)

    # Gradiente para calcular el Jacobiano
    dE_dY = -np.ones(E.shape)

    # Error para el Jacobiano (dJ_dY = Jac*E_Jac = dJ_dY*dE_dY*E_Jac)
    E_Jac = E

    return J, dJ_dY, dE_dY, E_Jac
