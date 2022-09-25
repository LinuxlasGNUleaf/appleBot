import math

import numpy as np

WORKER_MAX_ITERATIONS = 200

A = 2e6
BATTLE_FIELD_W: float = math.sqrt(A * 16 / 9)
BATTLE_FIELD_H: float = math.sqrt(A * 9 / 16)
NUM_PLANETS: int = 24
MAX_PLAYERS: int = 12
SEGMENT_STEPS: int = 25
MAX_SEGMENTS: int = 2000
MARGIN: int = 500

ENERGY_UPDATE_INTERVAL = 2
SCAN_INTERVAL = 10

# broad shot
B_SCAN_ANGLE_MARGIN = 60
B_SCAN_ANGLE_INC = 0.1
B_PLAYER_SIZE: float = 80
B_VELOCITY_RANGE = (10, 13, 1)

# fine shot
F_SCAN_ANGLE_MARGIN = 10
F_SCAN_ANGLE_INC = 0.01
F_PLAYER_SIZE: float = 4.0


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
