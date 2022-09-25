from concurrent.futures import ThreadPoolExecutor, as_completed

from numba import njit, double, int16, boolean

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

    def scan_range(self, angle_range, velocity_range, opponent_id, broad):
        tmp_angles = np.arange(*angle_range)
        print(f"calculating {len(tmp_angles) * (len(velocity_range) if broad else 1)} trajectories ")
        flag_array = np.asarray([0], dtype=bool)
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = []
            for velocity in (np.arange(*velocity_range) if broad else [velocity_range]):
                ang_i = 0
                while ang_i < len(tmp_angles):
                    angle_slice = tmp_angles[ang_i:ang_i + WORKER_MAX_ITERATIONS]

                    futures.append(ex.submit(scan_angle_f,
                                             planet_positions=self.planet_positions,
                                             planet_radii=self.planet_radii,
                                             planet_masses=self.planet_masses,
                                             start_position=self.player_positions[self.own_id],
                                             target_position=self.player_ids[opponent_id],
                                             angle_range=angle_slice,
                                             angle_count=len(angle_slice),
                                             velocity=velocity,
                                             broad=broad,
                                             flag_array=flag_array
                                             ))
                    ang_i += WORKER_MAX_ITERATIONS

            for completed_future in as_completed(futures):
                res = completed_future.result()
                if res[0] != 0:
                    print("Killing all remaining threads...")
                    flag_array[0] = True
                    ex.shutdown(wait=False, cancel_futures=True)
                    return res[1:]
                else:
                    print("No result from worker.")
            print("No angle found.")
            return None


@njit(nogil=True)
def scan_angle_f(planet_positions: np.ndarray(dtype=np.float64, shape=(NUM_PLANETS, 2)),
                 planet_radii: np.ndarray(dtype=np.float64, shape=(NUM_PLANETS,)),
                 planet_masses: np.ndarray(dtype=np.float64, shape=(NUM_PLANETS,)),
                 start_position: np.ndarray(dtype=np.float64, shape=(2,)),
                 target_position: np.ndarray(dtype=np.float64, shape=(2,)),
                 angle_range: np.ndarray(dtype=np.float64, shape=(1,)),
                 angle_count: int,
                 velocity: double,
                 broad: boolean,
                 flag_array: np.ndarray(dtype=bool, shape=(1,))
                 ) -> np.ndarray(dtype=np.float64, shape=(3,)):
    result = np.zeros(3)
    result[2] = velocity

    start_angle: float = 0
    hit_counter: int = 0

    for ang_i in range(angle_count):
        if flag_array[0]:
            return result
        start_angle = 0
        dist = simulate_shot_f(planet_positions=planet_positions,
                               planet_radii=planet_radii,
                               planet_masses=planet_masses,
                               start_position=start_position,
                               target_position=target_position,
                               angle=angle_range[ang_i],
                               velocity=velocity,
                               broad=broad
                               )

        if dist == 0:
            if hit_counter == 0:
                start_angle = angle_range[ang_i]
            hit_counter += 1

        elif hit_counter != 0:
            result[0] = hit_counter
            result[1] = (start_angle + angle_range[ang_i - 1]) / 2
            break
    return result


@njit(nogil=True)
def simulate_shot_f(planet_positions: np.ndarray(dtype=np.float64, shape=(NUM_PLANETS, 2)),
                    planet_radii: np.ndarray(dtype=np.float64, shape=(NUM_PLANETS, 1)),
                    planet_masses: np.ndarray(dtype=np.float64, shape=(NUM_PLANETS, 1)),
                    start_position: np.ndarray(dtype=np.float64, shape=(2,)),
                    target_position: np.ndarray(dtype=np.float64, shape=(2,)),
                    angle: double,
                    velocity: double,
                    broad: boolean
                    ) -> double:
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
                return min_distance

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
            if distance <= (B_PLAYER_SIZE if broad else F_PLAYER_SIZE):
                return 0
            elif self_distance <= F_PLAYER_SIZE:
                return math.inf

        elif self_distance > F_PLAYER_SIZE + 1.0:
            left_source = True

        # check if missile is out of bounds
        if position[0] < -MARGIN or \
                position[0] > BATTLE_FIELD_W + MARGIN or \
                position[1] < -MARGIN or \
                position[1] > BATTLE_FIELD_H + MARGIN:
            return min_distance

        # check if missile trail is too long
        if segment_count >= MAX_SEGMENTS:
            return min_distance
