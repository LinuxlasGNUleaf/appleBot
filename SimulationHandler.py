import math

import numpy as np
from numba import njit, double, int16
from concurrent.futures import ThreadPoolExecutor, as_completed

import utils

WORKER_MAX_ITERATIONS = 500

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

        self.player_positions: np.ndarray(dtype=np.float64, shape=(MAX_PLAYERS, 2)) = np.zeros((MAX_PLAYERS, 2))
        self.player_ids: np.ndarray(dtype=int, shape=(MAX_PLAYERS,)) = np.zeros((MAX_PLAYERS,), dtype=int)
        self.player_count: int = -1
        self.planet_positions: np.ndarray(dtype=np.float64, shape=(NUM_PLANETS, 2)) = np.zeros((NUM_PLANETS, 2))
        self.planet_radii: np.ndarray(dtype=np.float64, shape=(NUM_PLANETS, 0)) = np.zeros((NUM_PLANETS,))
        self.planet_masses: np.ndarray(dtype=np.float64, shape=(NUM_PLANETS, 0)) = np.zeros((NUM_PLANETS,))

    def set_field(self, planets: list[utils.Planet], players: dict[int, utils.Player], own_id: int):
        self.initialized = True

        self.position = players[own_id].position
        self.own_id = own_id

        # populate numpy arrays
        players = players.values()
        for i, player in enumerate(players):
            self.player_positions[i] = player.position
            self.player_ids[i] = player.id
        self.player_count = len(players)

        for i, planet in enumerate(planets):
            self.planet_positions[i] = planet.position
            self.planet_radii[i] = planet.radius
            self.planet_masses[i] = planet.mass

    def scan_range(self, angle_range, velocity_range):
        tmp_angles = np.arange(*angle_range)
        print(len(tmp_angles))
        with ThreadPoolExecutor(max_workers=16) as ex:
            futures = []
            for velocity in np.arange(*velocity_range):
                ang_i = 0
                while ang_i < len(tmp_angles):
                    angle_slice = tmp_angles[ang_i:ang_i + WORKER_MAX_ITERATIONS]
                    # print(f"starting worker with {len(angle_slice)} iterations.")
                    futures.append(ex.submit(scan_angle_f,
                                             planet_positions=self.planet_positions,
                                             planet_radii=self.planet_radii,
                                             planet_masses=self.planet_masses,
                                             player_positions=self.player_positions,
                                             player_ids=self.player_ids,
                                             player_count=self.player_count,
                                             position=self.position,
                                             angle_range=angle_slice,
                                             angle_count=len(angle_slice),
                                             velocity=velocity,
                                             own_id=self.own_id
                                             ))
                    ang_i += WORKER_MAX_ITERATIONS

            for completed_future in as_completed(futures):
                if completed_future.result()[0] != -1:
                    ex.shutdown(cancel_futures=True, wait=False)
                    return completed_future.result()
                else:
                    print("Worker yielded no solution.")
            else:
                return np.asarray([-1, 0, 0])



@njit(nogil=True)
def scan_angle_f(planet_positions: np.ndarray(dtype=np.float64, shape=(NUM_PLANETS, 2)),
                 planet_radii: np.ndarray(dtype=np.float64, shape=(NUM_PLANETS,)),
                 planet_masses: np.ndarray(dtype=np.float64, shape=(NUM_PLANETS,)),
                 player_positions: np.ndarray(dtype=np.float64, shape=(MAX_PLAYERS, 2)),
                 player_ids: np.ndarray(dtype=int, shape=(MAX_PLAYERS,)),
                 player_count: int16,
                 position: np.ndarray(dtype=np.float64, shape=(2,)),
                 angle_range: np.ndarray(dtype=np.float64, shape=(1,)),
                 angle_count: int,
                 velocity: double,
                 own_id: int16
                 ) -> int16:
    result = np.zeros(3)
    for ang_i in range(angle_count):
        player_id = simulate_shot_f(planet_positions=planet_positions,
                                    planet_radii=planet_radii,
                                    planet_masses=planet_masses,
                                    player_positions=player_positions,
                                    player_ids=player_ids,
                                    player_count=player_count,
                                    position=np.copy(position),
                                    angle=angle_range[ang_i],
                                    velocity=velocity,
                                    own_id=own_id
                                    )
        if player_id != -1:
            result[0] = player_id
            result[1] = angle_range[ang_i]
            result[2] = velocity
            break
    else:
        result[0] = -1
    return result


@njit(nogil=True)
def simulate_shot_f(planet_positions: np.ndarray(dtype=np.float64, shape=(NUM_PLANETS, 2)),
                    planet_radii: np.ndarray(dtype=np.float64, shape=(NUM_PLANETS, 1)),
                    planet_masses: np.ndarray(dtype=np.float64, shape=(NUM_PLANETS, 1)),
                    player_positions: np.ndarray(dtype=np.float64, shape=(MAX_PLAYERS, 2)),
                    player_ids: np.ndarray(dtype=int, shape=(MAX_PLAYERS, 1)),
                    player_count: int16,
                    position: np.ndarray(dtype=np.float64, shape=(1, 2)),
                    angle: double,
                    velocity: double,
                    own_id: int16
                    ) -> int16:
    speed = np.asarray([velocity * math.cos(math.radians(angle)),
                        velocity * -math.sin(math.radians(angle))],
                       dtype=np.float64)

    left_source: bool = False
    segment_count: int16 = 0
    while True:
        for i in range(NUM_PLANETS):
            # calculate vector from planet to missile and distance
            tmp_v = planet_positions[i] - position
            distance = np.linalg.norm(tmp_v)

            # collision with planet?
            if distance <= planet_radii[i]:
                return -1

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
                    return -1
                else:
                    return player_ids[i]

            if distance > PLAYER_SIZE + 1.0 and player_ids[i] == own_id:
                left_source = True

        # check if missile is out of bounds
        if position[0] < -MARGIN or \
                position[0] > BATTLE_FIELD_W + MARGIN or \
                position[1] < -MARGIN or \
                position[1] > BATTLE_FIELD_H + MARGIN:
            return -1

        # check if missile trail is too long
        if segment_count >= MAX_SEGMENTS:
            return -1
