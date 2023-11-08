import numpy as np
from libraries_.delta_hidden_ import delta_hidden_
from libraries_.delta_output import delta_output
from libraries_.delta_output_lm import delta_output_lm
from libraries_.model import Model
from libraries_.neural_layer_ import neural_layer


def Jacobian_k_opt_v2(Y, model, params, s):
    n_samples = len(Y)
    lambda_ = params.lambda_reg
    bias = np.ones((n_samples, 1))
    n_capas = len(params.W)

    model.dE_dW = [np.array([]) for _ in range(n_capas)]
    model.deltas = [np.array([]) for _ in range(n_capas)]

    Jac = np.zeros((params.nw_total, n_samples))

    # for k in range(n_capas - 1):
        # model.inputs[k + 1], _, _ = neural_layer(model.inputs[k], params.W[k], params.b[k], params.hidden_fun[k])



    # Capa de salida
    k = n_capas - 1
    model.deltas[k] = delta_output_lm(Y, model.inputs[k + 1], model.phi_[k], params)

    model.dE_dW[k] = np.reshape(model.deltas[k],(n_samples,1)) * np.concatenate((bias, model.inputs[k]), axis=1) + lambda_ * np.concatenate(
        ([0], params.W[k].flatten())) / n_samples
    Jac[(s) * (params.n_hidden[-1] + 1):(s+1) * (params.n_hidden[-1] + 1), :] = model.dE_dW[k].T


    # for k in range(n_capas - 2, -1, -1):
    #     model.deltas[k] = delta_hidden_(model.phi_[k], params.W[k + 1], model.deltas[k + 1])
    #
    #     nf, nc = params.W[k].shape
    #     idx0 = params.indx_W_Jac[k][0]
    #
    #     tmp = np.concatenate((bias, model.inputs[k]), axis=1)
    #
    #     for i in range(nf):
    #         Jac[idx0 + i * (nc + 1):idx0 + (i + 1) * (nc + 1), :] = (model.deltas[k][:, i] * tmp).T
    #
    # return Jac