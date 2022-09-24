from enum import Enum
import math

import numpy as np


class Player:
    def __init__(self, x, y, id):
        self.position = np.asarray([x, y], dtype=np.float64)
        self.id = id


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


class Missile:
    def __init__(self, pos, angle, energy):
        angle = math.radians(angle)

        self.speed = np.asarray([
            energy * math.cos(angle),
            energy * -math.sin(angle)
        ], dtype=np.float64)
        self.position = pos
        self.left_source = False


class MissileResult(Enum):
    RES_UNDETERMINED = -1
    RES_HIT_PLANET = 0
    RES_HIT_PLAYER = 1
    RES_OUT_OF_BOUNDS = 2
    RES_OUT_OF_SEGMENTS = 3
