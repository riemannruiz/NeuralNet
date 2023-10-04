def delta_hidden_(phi_, W, d_1):
    d = (d_1 @ W) * phi_ #calculo de \delta_{1}
    return d