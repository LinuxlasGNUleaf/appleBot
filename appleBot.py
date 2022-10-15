import math
import random
from datetime import datetime

from SimulationHandler import SimulationHandler
from utils import *


class AppleBot:
    def __init__(self, socket_manager):
        self.connection = socket_manager
        self.simulation = SimulationHandler(self)

        self.id = -1
        self.name = self.__class__.__name__
        self.planets = []
        self.players = {}
        self.opponents = []
        self.ignored_opponents = []
        self.angle = 0
        self.speed = 10
        self.energy = 0

        self.last_name_update = datetime.fromtimestamp(0)

        self.update_flag = False
        self.ignore_msg_sent = False

        self.init()

    def init(self):
        self.show_state("IDLING")

    def show_state(self, state):
        if state == "IDLING":
            state = "-"
        elif state == "SCANNING":
            state = "0"
        elif state == "TARGETING":
            state = "X"
        else:
            state = "[???]"

        self.connection.send_str(f"n {self.name} [{state}]")

    def msg(self, message):
        print(f"[{datetime.now()}] [{self.__class__.__name__}]: {message}")

    def shoot(self):
        self.angle += 361 / 36.0
        if self.angle > 180:
            self.angle -= 360
        return self.speed, self.angle

    def update_simulation(self):
        if self.id == -1:
            return
        if self.id not in self.players:
            return
        if not self.planets:
            return
        if not self.opponents:
            return
        self.msg("Updating simulation...")
        self.simulation.update_field()

    def process_incoming(self):
        struct_data = self.connection.receive_struct("II")

        # recv timed out
        if struct_data is None:
            return False

        msg_type, payload = struct_data

        # bot has joined
        if msg_type == 1:
            self.id = payload
            self.msg(f"set id to {payload}")
            self.update_flag = True

        # player left
        elif msg_type == 2:
            self.msg(f"Player {payload} left.")
            del self.players[payload]
            self.opponents.remove(payload)
            if payload in self.ignored_opponents:
                self.ignored_opponents.remove(payload)
            self.update_flag = True

        # player joined/reset
        elif msg_type == 3:
            x, y = self.connection.receive_struct("ff")
            if payload not in self.players:
                self.msg(f"player {payload} joined the game at ({round(x)},{round(y)})")
                if payload != self.id:
                    self.opponents.append(payload)
                new_player = Player(x, y, payload)

            else:
                self.msg(f"player {payload} moved to ({round(x)},{round(y)})")
                if payload in self.ignored_opponents:
                    self.ignored_opponents.remove(payload)
                elif payload == self.id:
                    self.ignored_opponents.clear()
                new_player = self.players[payload]
                new_player.position[0] = x
                new_player.position[1] = y

            self.players[payload] = new_player
            self.update_flag = True

        # shot finished msg, deprecated
        elif msg_type == 4:
            self.msg("WARNING! WRONG BOT PROTOCOL VERSION DETECTED (VER < 8)! THIS MSG_TYPE SHOULD NOT BE RECEIVED!")
            exit(1)

        # shot begin
        elif msg_type == 5:
            _, _ = self.connection.receive_struct("dd")

        # shot end (discard shot data)
        elif msg_type == 6:
            _, _, length = self.connection.receive_struct("ddI")
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
            self.msg(f"map data changed. {payload} planets received.")
            self.update_flag = True

        # unknown MSG_TYPE
        else:
            self.msg(f"Unexpected message_type: '{msg_type}'\n\t- data: '{payload}'")
            return False

        return True

    def loop(self):
        if self.process_incoming():
            return

        self.scan_field()

    def scan_field(self):
        possible_targets = list(set(self.opponents).difference(set(self.ignored_opponents)))

        # No viable opponents found to target (any opponents still on the board haven't moved since last scan)
        if not possible_targets:
            if not self.ignore_msg_sent:
                self.msg("Idling due to no viable opponents.")
                self.ignore_msg_sent = True
            return
        self.ignore_msg_sent = False
        # update simulation field if flag is set
        if self.update_flag:
            self.msg("Update flag set, updating field...")
            self.simulation.update_field()
            self.update_flag = False
        # if field is not ready yet, return
        elif not self.simulation.initialized:
            return

        target_player = random.choice(possible_targets)
        result = self.simulation.scan_for(target_player)
        # field changed
        if result == -2:
            return
        # target parameters found
        elif result != -1:
            self.connection.send_str(f"c\nv {result[1]}")
            self.connection.send_str(f"{math.degrees(result[0])}")
        self.ignored_opponents.append(target_player)

        self.show_state("IDLING")
