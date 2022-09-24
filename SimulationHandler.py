import math
import numpy as np
from tqdm import tqdm
import utils


class SimulationHandler:
    def __init__(self):
        self.A = 2e6
        self.battlefieldW = math.sqrt(self.A * 16 / 9)
        self.battlefieldH = math.sqrt(self.A * 9 / 16)
        self.playerSize = 4.0
        self.margin = 500
        self.players = None
        self.planets = None
        self.maxSegments = 2000
        self.segmentSteps = 25
        self.maxPlayers = 12
        self.numPlanets = 24

        self.own_id = None
        self.position = None
        self.initialized = False

    def set_field(self, planets, players, own_id):
        self.initialized = True
        self.players = players
        self.planets = planets
        self.own_id = own_id

        for player in players:
            if player.id == own_id:
                self.position = player.position

    def simulate_own_shot(self, angle, energy):
        missile = utils.Missile(np.copy(self.position), energy, angle)
        missile_trace = []
        missile_result = utils.MissileResult.RES_UNDETERMINED
        info = 0
        sim_running = True

        while sim_running:
            for planet in self.planets:
                # calculate vector from planet to missile and distance
                tmp_v = planet.position - missile.position
                distance = np.linalg.norm(tmp_v)

                # collision with planet?
                if distance <= planet.radius:
                    missile_result = utils.MissileResult.RES_HIT_PLANET
                    info = planet.id
                    sim_running = False
                    break

                # normalize tmp vector
                tmp_v /= distance
                # apply Newtonian Gravity
                tmp_v *= planet.mass / (distance ** 2)
                # shortening to segment
                tmp_v /= self.segmentSteps

                # add tmp vector to missile speed vector
                missile.speed += tmp_v

            if not sim_running:
                break

            # shortening resulting speed to segment
            tmp_v = missile.speed / self.segmentSteps
            # apply speed vector
            missile.position += tmp_v

            # check if missile hit a player
            for player in self.players:
                distance = np.linalg.norm(player.position - missile.position)

                if distance <= self.playerSize and missile.left_source:
                    missile_result = utils.MissileResult.RES_HIT_PLAYER
                    info = player.id
                    sim_running = False
                    break

                if distance > self.playerSize + 1.0 and player.id == self.own_id:
                    missile.left_source = True

            if not sim_running:
                break

            # check if missile is out of bounds
            if missile.position[0] < -self.margin or \
                    missile.position[0] > self.battlefieldW + self.margin or \
                    missile.position[1] < -self.margin or \
                    missile.position[1] > self.battlefieldH + self.margin:
                missile_result = utils.MissileResult.RES_OUT_OF_BOUNDS
                break

            missile_trace.append(missile.position)
            if len(missile_trace) >= self.maxSegments:
                missile_result = utils.MissileResult.RES_OUT_OF_SEGMENTS
                break

        return missile_result, info, missile_trace

    def scan_angle(self, angle_arange, energy_arange):
        results = []
        for energy in tqdm(np.arange(*energy_arange), leave=False, colour='green', desc='energy:'):
            for angle in tqdm(np.arange(*angle_arange), leave=False, colour='red', desc='angle:'):
                res, info, trace = self.simulate_own_shot(angle, energy)
                if res == utils.MissileResult.RES_HIT_PLAYER and info != self.own_id:
                    results.append((info, len(trace), angle, energy))

        print(results)
