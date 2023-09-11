
def delta_hidden(phi_, W, d_1):
    d = (d_1 @ W.T) * phi_ #calculo de \delta_{1}
    return d