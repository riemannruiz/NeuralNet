def delta_output(Y, Yhat,phi_,params):
    _, dJ_dY,_,_ = params.loss(Y, Yhat, params)

    d = phi_ * dJ_dY

    return d
