import sys

from numba import njit, double, int16, boolean, prange
from utils import *


class SimulationHandler:
    def __init__(self):
        self.own_id: int = -1
        self.initialized = False

        self.player_positions: np.ndarray(dtype=np.float64, shape=(MAX_PLAYERS, 2)) = np.zeros((MAX_PLAYERS, 2))
        self.player_ids: np.ndarray(dtype=int, shape=(MAX_PLAYERS,)) = np.zeros((MAX_PLAYERS,), dtype=int)
        self.player_count: int = -1
        self.planet_positions: np.ndarray(dtype=np.float64, shape=(NUM_PLANETS, 2)) = np.zeros((NUM_PLANETS, 2))
        self.planet_radii: np.ndarray(dtype=np.float64, shape=(NUM_PLANETS, 0)) = np.zeros((NUM_PLANETS,))
        self.planet_masses: np.ndarray(dtype=np.float64, shape=(NUM_PLANETS, 0)) = np.zeros((NUM_PLANETS,))

    def set_field(self, planets: list[Planet], players: dict[int, Player], own_id: int):
        self.initialized = True

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

    def scan_for(self, target_id):
        angle_list = np.linspace(0, 360, 10)
        print(angle_list.size)
        results = np.zeros(dtype=np.float64, shape=(angle_list.size,))
        scan_list(planet_positions=self.planet_positions,
                  planet_radii=self.planet_radii,
                  planet_masses=self.planet_masses,
                  start_position=self.player_positions[self.own_id],
                  target_position=self.player_positions[target_id],
                  angle_list=angle_list,
                  angle_count=angle_list.size,
                  velocity=10,
                  results=results)
        sys.stdout.write("[")
        for i in results[:-1]:
            sys.stdout.write(f"{round(i)},")
        sys.stdout.write(f"{round(results[-1])}]\n[")
        for i in angle_list[:-1]:
            sys.stdout.write(f"{round(i)},")
        sys.stdout.write(f"{round(angle_list[-1])}]\n")
        best_index = np.where(results == min(results))[0][0]
        best_angle = angle_list[best_index]
        print(f"Best results found for angle {round(best_angle,2)}°, where the distance was: {round(results[best_index])}")
        print(best_angle)

        return best_angle


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
            if distance <= F_PLAYER_SIZE:
                results[index] = 0
                return
            elif self_distance <= F_PLAYER_SIZE:
                results[index] = math.inf
                return

        elif self_distance > F_PLAYER_SIZE + 1.0:
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
