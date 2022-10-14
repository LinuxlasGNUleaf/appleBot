import math
import numpy as np

A = 2e6
BATTLE_FIELD_W: float = math.sqrt(A * 16 / 9)
BATTLE_FIELD_H: float = math.sqrt(A * 9 / 16)
F_PLAYER_SIZE: float = 4.0
MARGIN: int = 500

NUM_PLANETS: int = 24
MAX_PLAYERS: int = 12
MAX_SEGMENTS: int = 2000
SEGMENT_STEPS: int = 25

ENERGY_UPDATE_INTERVAL = 2
SCAN_INTERVAL = 10


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
