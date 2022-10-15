import numpy as np


class Player:
    def __init__(self, x, y, pid):
        self.position = np.asarray([x, y], dtype=np.float64)
        self.id = pid


class Curve:
    def __init__(self, n, curves):
        self.n = n
        self.curves = curves


class Planet:
    def __init__(self, x, y, radius, mass, pid):
        self.position = np.asarray([x, y], dtype=np.float64)
        self.mass = mass
        self.radius = radius
        self.id = pid
