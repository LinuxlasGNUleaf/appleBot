import math
import random
from datetime import datetime

from SimulationHandler import SimulationHandler
from utils import *

ENERGY_UPDATE_INTERVAL = 2
SCAN_INTERVAL = 10


class AppleBot:
    def __init__(self, socket_manager):
        self.connection = socket_manager
        self.simulation = SimulationHandler()

        self.id = -1
        self.name = self.__class__.__name__
        self.planets = []
        self.players = {}
        self.opponent_ids = []
        self.angle = 0
        self.speed = 10
        self.last_shot = []
        self.energy = 0

        self.last_energy_update = datetime.fromtimestamp(0)
        self.last_scan = datetime.fromtimestamp(0)

        self.init()

    def init(self):
        self.connection.send_str(f"n {self.name}")

    def msg(self, message):
        print(f"[{self.__class__.__name__}]: {message}")

    def shoot(self):
        self.angle += 361 / 36.0
        if self.angle > 180:
            self.angle -= 360
        return self.speed, self.angle

    def report_shot(self, curve):
        self.last_shot = curve

    def update_simulation(self):
        if self.id == -1:
            return
        if not self.planets:
            return
        if not self.players:
            return
        self.simulation.set_field(self.planets, self.players, self.id)

    def process_incoming(self):
        struct_data = self.connection.receive_struct("II")

        # recv timed out
        if struct_data is None:
            return

        msg_type, payload = struct_data

        # bot has joined
        if msg_type == 1:
            self.id = payload
            self.msg(f"set id to {payload}")
            self.update_simulation()

        # player left
        elif msg_type == 2:
            del self.players[payload]
            self.opponent_ids.remove(payload)
            self.update_simulation()

        # player joined/reset
        elif msg_type == 3:
            x, y = self.connection.receive_struct("ff")

            if payload not in self.players:
                self.msg(f"player {payload} joined the game at ({round(x)},{round(y)})")
                if id != self.id:
                    self.opponent_ids.append(payload)
                new_player = Player(x, y, payload)

            else:
                self.msg(f"player {payload} moved to ({round(x)},{round(y)})")
                new_player = self.players[payload]
                new_player.position[0] = x
                new_player.position[1] = y

            self.players[payload] = new_player
            self.update_simulation()

        # shot finished msg, deprecated
        elif msg_type == 4:
            self.msg("WARNING! WRONG BOT PROTOCOL VERSION DETECTED (VER < 8)! THIS MSG_TYPE SHOULD NOT BE RECEIVED!")
            exit(1)

        # shot begin
        elif msg_type == 5:
            angle, velocity = self.connection.receive_struct("dd")
            self.msg(f"player {payload} launched a missile with angle {round(angle, 3)}° and velocity {velocity}")

        # shot end (discard shot data)
        elif msg_type == 6:
            angle, velocity, length = self.connection.receive_struct("ddI")
            for i in range(length):
                _ = self.connection.receive_struct("ff")

        # game mode, deprecated
        elif msg_type == 7:
            self.msg("WARNING! WRONG BOT PROTOCOL VERSION DETECTED (VER <= 8)! THIS MSG_TYPE SHOULD NOT BE RECEIVED!")
            exit(1)

        # own energy
        elif msg_type == 8:
            self.energy = math.floor(self.connection.receive_struct("d")[0])

        # planet pos
        elif msg_type == 9:
            # discard planet byte count
            self.connection.receive_struct("I")

            self.planets = []
            for i in range(payload):
                x, y, radius, mass = self.connection.receive_struct("dddd")
                self.planets.append(Planet(x, y, radius, mass, i))
            self.msg(f"planet data for {len(self.planets)} planets received")
            self.update_simulation()

        # unknown MSG_TYPE
        else:
            self.msg(f"Unexpected message_type: '{msg_type}'\n\t- data: '{payload}'")

    def loop(self):

        if (datetime.now() - self.last_energy_update).seconds > ENERGY_UPDATE_INTERVAL:
            self.connection.send_str("u")
            self.last_energy_update = datetime.now()

        if (datetime.now() - self.last_scan).seconds > SCAN_INTERVAL:
            self.scan_field()

        self.process_incoming()

    def scan_field(self):
        if len(self.players) <= 1:
            return
        if not self.simulation.initialized:
            return

        target_player = self.players[random.choice(self.opponent_ids)]
        source_player = self.players[self.id]
        print((target_player.position[0]-source_player.position[0], target_player.position[1]-source_player.position[1]))
        target_angle = math.atan2(target_player.position[1]-source_player.position[1],
                                  target_player.position[0]-source_player.position[0])
        print(f"target angle is: {math.degrees(target_angle)}°")

        self.simulation.scan_angle((0, 360, 0.5), (9, 12, 1))
        self.last_scan = datetime.now()
