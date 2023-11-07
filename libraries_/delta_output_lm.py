def delta_output_lm(Y, Yhat, phi_, params):
    _, _, dE_dY, _ = params.loss(Y, Yhat, params)
    d = phi_ * dE_dY
    return d