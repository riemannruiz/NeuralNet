
class Model:
    def __init__(self):
        self.inputs = []  # Lista de arrays numpy
        self.hidden_out = []  # Lista de arrays numpy
        self.phi_ = []  # Lista de arrays numpy
        self.loss = []  # Lista de celdas
        self.deltas = []  # Lista de arrays numpy
        self.b_gradient = []  # Lista de arrays numpy
        self.W_gradient = []  # Lista de arrays numpy
        self.d_b = []  # Lista de arrays numpy
        self.d_W = []  # Lista de arrays numpy

