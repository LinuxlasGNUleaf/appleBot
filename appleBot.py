import math
import random
from datetime import datetime

from SimulationHandler import SimulationHandler
from utils import *


class AppleBot:
    def __init__(self, socket_manager):
        self.connection = socket_manager
        self.simulation = SimulationHandler()

        self.id = -1
        self.name = self.__class__.__name__
        self.planets = []
        self.players = {}
        self.ignored_ids = []
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
        if self.id not in self.players:
            return
        if not self.planets:
            return
        if not self.opponent_ids:
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
                if payload != self.id:
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
        if not self.opponent_ids:
            return
        if not self.simulation.initialized:
            return

        target_player = self.players[random.choice(self.opponent_ids)]
        source_player = self.players[self.id]

        diff_y = target_player.position[1] - source_player.position[1]
        diff_x = target_player.position[0] - source_player.position[0]
        target_angle = math.degrees(math.atan2(-diff_y, diff_x))
        print(f"target angle: {target_angle}°")

        print("starting broad simulation...")
        angle_range = (math.radians(target_angle - B_SCAN_ANGLE_MARGIN),
                       math.radians(target_angle + B_SCAN_ANGLE_MARGIN),
                       math.radians(B_SCAN_ANGLE_INC)
                       )
        print(", ".join([str(round(math.degrees(angle), 1)) for angle in angle_range]))
        res = self.simulation.scan_range(angle_range, B_VELOCITY_RANGE, target_player.id, broad=True)

        if res is None:
            print("No trajectory to this player found.")
            self.last_scan = datetime.now()
            return

        print(f"broad trajectory to player {target_player.id} found (angle: {math.degrees(res[0])}°). starting accurate simulation...")
        self.connection.send_str(f"v {res[1]}")
        self.connection.send_str(f"{math.degrees(res[0])}")
        angle_range = (res[0] - math.radians(F_SCAN_ANGLE_MARGIN),
                       res[0] + math.radians(F_SCAN_ANGLE_MARGIN),
                       math.radians(F_SCAN_ANGLE_INC)
                       )
        print(", ".join([str(round(math.degrees(angle), 1)) for angle in angle_range]))

        res = self.simulation.scan_range(angle_range, res[1], target_player.id, broad=False)
        if res is None:
            print("No angle found.")
            self.last_scan = datetime.now()
            return

        print(f"accurate trajectory to player {target_player.id} found!")
        print(res)
        self.connection.send_str(f"v {res[1]}")
        self.connection.send_str(f"{math.degrees(res[0])}")
