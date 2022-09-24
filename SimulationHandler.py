import math

import numpy as np
import scipy
from tqdm import tqdm
from numba import njit, double, int16

import utils

A = 2e6
BATTLE_FIELD_W: float = math.sqrt(A * 16 / 9)
BATTLE_FIELD_H: float = math.sqrt(A * 9 / 16)
NUM_PLANETS: int = 24
MAX_PLAYERS: int = 12
SEGMENT_STEPS: int = 25
MAX_SEGMENTS: int = 2000
PLAYER_SIZE: float = 4.0
MARGIN: int = 500


class SimulationHandler:
    def __init__(self):
        self.own_id: int = -1
        self.position: np.ndarray(dtype=np.float64, shape=(1, 2)) = None
        self.initialized = False

        self.player_positions: np.ndarray(dtype=np.float64, shape=(MAX_PLAYERS, 2)) = None
        self.player_ids: np.ndarray(dtype=int, shape=(MAX_PLAYERS, 1)) = None
        self.player_count: int = -1
        self.planet_positions: np.ndarray(dtype=np.float64, shape=(NUM_PLANETS, 2)) = None
        self.planet_radii: np.ndarray(dtype=np.float64, shape=(NUM_PLANETS, 1)) = None
        self.planet_masses: np.ndarray(dtype=np.float64, shape=(NUM_PLANETS, 1)) = None

    def set_field(self, planets: list[utils.Planet], players: dict[int, utils.Player], own_id: int):
        self.initialized = True

        self.position = players[own_id].position
        self.own_id = own_id

        # populate numpy arrays
        players = players.values()
        self.player_positions = np.asarray([player.position for player in players])
        self.player_ids = np.asarray([player.id for player in players])
        self.player_count = len(players)
        self.planet_positions = np.asarray([planet.position for planet in planets])
        self.planet_radii = np.asarray([planet.radius for planet in planets])
        self.planet_masses = np.asarray([planet.mass for planet in planets])

    def scan_angle(self, angle_range, velocity_range):
        results = []
        for velocity in tqdm(np.arange(*velocity_range), leave=False, colour='green', desc='velocity:'):
            for angle in tqdm(np.arange(*angle_range), leave=False, colour='red', desc='angle:'):
                res = np.zeros((3, 1))
                simulate_shot_f(planet_positions=self.planet_positions,
                                planet_radii=self.planet_radii,
                                planet_masses=self.planet_masses,
                                player_positions=self.player_positions,
                                player_ids=self.player_ids,
                                player_count=self.player_count,
                                position=np.copy(self.position),
                                angle=angle,
                                velocity=velocity,
                                own_id=self.own_id,
                                result=res
                                )
                if res[0] != -1:
                    results.append(res)
                pass
        print(results)


@njit
def simulate_shot_f(planet_positions: np.ndarray(dtype=np.float64, shape=(NUM_PLANETS, 2)),
                    planet_radii: np.ndarray(dtype=np.float64, shape=(NUM_PLANETS, 1)),
                    planet_masses: np.ndarray(dtype=np.float64, shape=(NUM_PLANETS, 1)),
                    player_positions: np.ndarray(dtype=np.float64, shape=(MAX_PLAYERS, 2)),
                    player_ids: np.ndarray(dtype=int, shape=(MAX_PLAYERS, 1)),
                    player_count: int16,
                    position: np.ndarray(dtype=np.float64, shape=(1, 2)),
                    angle: double,
                    velocity: double,
                    own_id: int16,
                    result: np.ndarray(dtype=np.float64, shape=(3, 1))
                    ) -> None:
    speed = np.asarray([velocity * math.cos(angle),
                        velocity * -math.sin(angle)],
                       dtype=np.float64)

    result[0] = -1
    left_source: bool = False
    segment_count: int16 = 0
    while True:
        for i in range(NUM_PLANETS):
            # calculate vector from planet to missile and distance
            tmp_v = planet_positions[i] - position
            distance = np.linalg.norm(tmp_v)

            # collision with planet?
            if distance <= planet_radii[i]:
                return

            # normalize tmp vector
            tmp_v /= distance
            # apply Newtonian Gravity
            tmp_v *= planet_masses[i] / (distance ** 2)
            # shortening to segment
            tmp_v /= SEGMENT_STEPS

            # add tmp vector to missile speed vector
            speed += tmp_v

        # shortening resulting speed to segment
        tmp_v = speed / SEGMENT_STEPS
        # apply speed vector
        position += tmp_v

        # check if missile hit a player
        for i in range(player_count):
            distance = np.linalg.norm(player_positions[i] - position)

            if distance <= PLAYER_SIZE and left_source:
                if player_ids[i] == own_id:
                    return
                else:
                    result[0] = player_ids[i]
                    result[1] = angle
                    result[2] = velocity

            if distance > PLAYER_SIZE + 1.0 and player_ids[i] == own_id:
                left_source = True

        # check if missile is out of bounds
        if position[0] < -MARGIN or \
                position[0] > BATTLE_FIELD_W + MARGIN or \
                position[1] < -MARGIN or \
                position[1] > BATTLE_FIELD_H + MARGIN:
            return

        # check if missile trail is too long
        if segment_count >= MAX_SEGMENTS:
            return
