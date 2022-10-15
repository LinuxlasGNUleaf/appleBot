import math
import sys
from datetime import datetime

from numba import njit, double, int16, prange
from utils import *

A = 2e6
BATTLE_FIELD_W: float = math.sqrt(A * 16 / 9)
BATTLE_FIELD_H: float = math.sqrt(A * 9 / 16)
PLAYER_SIZE: float = 4
MARGIN: int = 500

NUM_PLANETS: int = 24
MAX_PLAYERS: int = 12
MAX_SEGMENTS: int = 2000
SEGMENT_STEPS: int = 25

ENERGY_UPDATE_INTERVAL = 2

BROAD_STEPS = 120
BROAD_TEST_CANDIDATES = 5
BROAD_DISTANCE_MAX = 20

FINE_STEPS = 50
VELOCITY_RANGE = [12, 13, 11, 14]


class SimulationHandler:
    def __init__(self, bot):
        self.bot = bot
        self.initialized = False

        self.player_positions: np.ndarray(dtype=np.float64, shape=(MAX_PLAYERS, 2)) = \
            np.zeros((MAX_PLAYERS, 2))
        self.planet_positions: np.ndarray(dtype=np.float64, shape=(NUM_PLANETS, 2)) = \
            np.zeros((NUM_PLANETS, 2))
        self.planet_radii: np.ndarray(dtype=np.float64, shape=(NUM_PLANETS, 0)) = \
            np.zeros((NUM_PLANETS,))
        self.planet_masses: np.ndarray(dtype=np.float64, shape=(NUM_PLANETS, 0)) = \
            np.zeros((NUM_PLANETS,))

    def msg(self, message):
        print(f"[{datetime.now()}] [{self.__class__.__name__}]: {message}")

    def update_field(self):
        self.initialized = True

        # populate numpy arrays
        for pid in range(MAX_PLAYERS):
            if pid in self.bot.players:
                self.player_positions[pid] = self.bot.players[pid].position
            else:
                self.player_positions[pid] = np.zeros(dtype=np.float64, shape=(2,))

        for i, planet in enumerate(self.bot.planets):
            self.planet_positions[i] = planet.position
            self.planet_radii[i] = planet.radius
            self.planet_masses[i] = planet.mass

    def run_scanlist(self, target_id, angle_list, velocity):
        results = np.full(dtype=np.float64, shape=(angle_list.size,), fill_value=math.inf)
        scan_list(planet_positions=self.planet_positions,
                  planet_radii=self.planet_radii,
                  planet_masses=self.planet_masses,
                  start_position=self.player_positions[self.bot.id],
                  target_position=self.player_positions[target_id],
                  angle_list=angle_list,
                  angle_count=angle_list.size,
                  velocity=velocity,
                  results=results)
        return results

    def check_for_relevant_update(self, target_id):
        self_pos = self.player_positions[self.bot.id].copy()
        target_pos = self.player_positions[target_id].copy()
        planet_pos = self.planet_positions.copy()
        while self.bot.process_incoming():
            pass
        self.update_field()
        if (self_pos != self.player_positions[self.bot.id]).all():
            return True
        if (target_pos != self.player_positions[target_id]).all():
            return True
        if (planet_pos != self.planet_positions).all():
            return True
        return False

    def scan_for(self, target_id):
        self.msg(f"======> SCANNING FOR PLAYER {target_id}")
        for velocity in VELOCITY_RANGE:
            self.bot.show_state("SCANNING")
            # broad scan
            sys.stdout.write(f"[{datetime.now()}] [{self.__class__.__name__}]: Scanning with velocity {velocity}...")

            # create angle list, remove last angle to avoid doubling 0 / 360°
            angle_list = np.linspace(0, 2 * math.pi, BROAD_STEPS + 1)[:-1]
            # run sim
            results = self.run_scanlist(target_id, angle_list, velocity)

            sorted_angles = angle_list[results.argsort()][:BROAD_TEST_CANDIDATES]
            sorted_angles = sorted_angles[sorted_angles < BROAD_DISTANCE_MAX]

            if sorted_angles.size == 0:
                sys.stdout.write("fail.\n")
                self.msg("No viable angles found at this velocity.")
                continue
            else:
                sys.stdout.write("success.\n")

            if self.check_for_relevant_update(target_id):
                self.msg("Situation changed, aborting simulation.")
                return -2  # field changed

            self.msg(f"Exploring {len(sorted_angles)} viable angles:")
            self.bot.show_state("TARGETING")

            for test_angle in sorted_angles:
                sys.stdout.write(
                    f"[{datetime.now()}] [{self.__class__.__name__}]: Exploring angle {round(math.degrees(test_angle), 2):05}°...")
                # create angle list, remove last angle to avoid doubling 0 / 360°
                angle_range = 2 * math.pi / BROAD_STEPS
                angle_list = np.linspace(test_angle - angle_range,
                                         test_angle + angle_range,
                                         FINE_STEPS + 1)[:-1]
                results = self.run_scanlist(target_id, angle_list, velocity)

                if self.check_for_relevant_update(target_id):
                    sys.stdout.write("fail.\n")
                    self.msg("Relevant information changed, aborting simulation.")
                    return -2  # field changed

                if (results < PLAYER_SIZE).any():
                    selected_index = np.where(results < PLAYER_SIZE)[0][0]
                    found_angle = angle_list[selected_index]
                    sys.stdout.write(f"success.\n")
                    self.msg(f"======> TARGET PARAMS: {round(math.degrees(found_angle), 2)}°, {velocity}")
                    return found_angle, velocity
                else:
                    sys.stdout.write("fail.\n")

            self.msg("No viable angles found at this velocity.")

        self.msg("======> NO ANGLES FOUND. ABORTING.")
        return -1


@njit(nogil=True, parallel=True)
def scan_list(planet_positions: np.ndarray(dtype=np.float64, shape=(NUM_PLANETS, 2)),
              planet_radii: np.ndarray(dtype=np.float64, shape=(NUM_PLANETS,)),
              planet_masses: np.ndarray(dtype=np.float64, shape=(NUM_PLANETS,)),
              start_position: np.ndarray(dtype=np.float64, shape=(2,)),
              target_position: np.ndarray(dtype=np.float64, shape=(2,)),
              angle_list: np.ndarray,
              angle_count: int,
              velocity: double,
              results: np.ndarray):
    for index in prange(angle_count):
        simulate_shot(planet_positions=planet_positions,
                      planet_radii=planet_radii,
                      planet_masses=planet_masses,
                      start_position=start_position,
                      target_position=target_position,
                      angle=angle_list[index],
                      velocity=velocity,
                      index=index,
                      results=results)


@njit(nogil=True)
def simulate_shot(planet_positions: np.ndarray(dtype=np.float64, shape=(NUM_PLANETS, 2)),
                  planet_radii: np.ndarray(dtype=np.float64, shape=(NUM_PLANETS,)),
                  planet_masses: np.ndarray(dtype=np.float64, shape=(NUM_PLANETS,)),
                  start_position: np.ndarray(dtype=np.float64, shape=(2,)),
                  target_position: np.ndarray(dtype=np.float64, shape=(2,)),
                  angle: double,
                  velocity: double,
                  index: int,
                  results: np.ndarray
                  ):
    position = start_position.copy()
    speed = np.asarray([velocity * math.cos(angle),
                        velocity * -math.sin(angle)],
                       dtype=np.float64)

    left_source: bool = False
    segment_count: int16 = 0
    min_distance: float = np.linalg.norm(target_position - start_position)

    while True:
        for i in range(NUM_PLANETS):
            # calculate vector and distance from planet to missile
            tmp_v = planet_positions[i] - position
            distance = np.linalg.norm(tmp_v)

            # collision with planet?
            if distance <= planet_radii[i]:
                results[index] = min_distance
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

        # check if missile hit target player
        distance = np.linalg.norm(target_position - position)
        self_distance = np.linalg.norm(target_position - start_position)

        min_distance = min(distance, min_distance)
        if left_source:
            if distance <= PLAYER_SIZE:
                results[index] = distance
                return
            elif self_distance <= PLAYER_SIZE:
                results[index] = math.inf
                return

        elif self_distance > PLAYER_SIZE + 1.0:
            left_source = True

        # check if missile is out of bounds
        if position[0] < -MARGIN or \
                position[0] > BATTLE_FIELD_W + MARGIN or \
                position[1] < -MARGIN or \
                position[1] > BATTLE_FIELD_H + MARGIN:
            results[index] = min_distance
            return

        # check if missile trail is too long
        if segment_count >= MAX_SEGMENTS:
            results[index] = min_distance
            return
