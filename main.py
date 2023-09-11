import numpy as np
from libraries_.f_sigmoid import f_sigmoid
from libraries_.neural_model_ import neural_model
from libraries_.f_lineal import f_lineal
from libraries_.ini_network import ini_network
from libraries_.cost_fun_rmse_reg_ import cost_fun_rmse_reg_
# from libraries_.Grad_Desc_training_ import Grad_Desc_training

# Generar los datos para el aprendizaje de la red neuronal


x1 = np.arange(-1, 1.01, 0.01)
y = 4 * x1 * np.sin(10 * x1) + x1 * np.exp(x1) * np.cos(20 * x1)
X = x1.reshape(-1, 1)
Y = y.reshape(-1, 1)

np.random.seed(1)

params = ini_network(X, Y, [10,5])
params.hidden_fun[0] = f_sigmoid
params.eta = 0.1
Yhat,_ = neural_model(X, params)
# print(Yhat.shape) (201,1)


J = cost_fun_rmse_reg_(Y,Yhat,params)

n_iter = 5000
#params, model = Grad_Desc_training(X, Y, params, n_iter)

# Obtener la segunda dimensión de la primera matriz en params.W

