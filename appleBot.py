import logging
import math
import random
import time

from SimulationHandler import SimulationHandler
from utils import *

NAMES = [
    "Appbomination",
    "Belta Lowda",
    "Marco Inaros",
    "Aegis",
    "Black Hole",
    "Newton's Bane",
    "Waymaker",
    "Gatecrasher",
    "Unmakyr",
    "RED/BLUE PILL",
    "Chicxulub",
    "Seeker"]


class AppleBot:
    def __init__(self, socket_manager):
        self.logger = logging.getLogger(__name__)

        # Set up SimulationHandler and precompile functions
        self.simulation = SimulationHandler(self)
        self.simulation.compile_functions()

        # Set uo ConnectionHandler and initialize connection
        self.connection = socket_manager
        self.connection.initialize()

        self.id = -1
        self.name = ""
        self.planets = []
        self.players = {}
        self.opponents = []
        self.ignored_opponents = []
        self.energy = 0

        self.last_name_update = time.time()
        self.update_flag = False

    def update_simulation(self):
        if self.id == -1:
            return
        if self.id not in self.players:
            return
        if not self.planets:
            return
        if not self.opponents:
            return
        self.logger.info("Updating simulation...")
        self.simulation.update_field()

    def choose_name(self):
        pick_names = NAMES.copy()
        try:
            pick_names.remove(self.name)
        except ValueError:
            pass
        name = random.choice(pick_names)
        self.connection.send_str(f"n {name}")

    def process_incoming(self, log=False):
        struct_data = self.connection.receive_struct("II")

        # recv timed out
        if struct_data is None:
            return False

        # unpack struct
        msg_type, payload = struct_data

        # bot has joined
        if msg_type == 1:
            self.id = payload
            self.update_flag = True
            if log:
                self.logger.info(f"RECV: Own id set to {payload}")

        # player left
        elif msg_type == 2:
            del self.players[payload]
            self.opponents.remove(payload)
            if payload in self.ignored_opponents:
                self.ignored_opponents.remove(payload)
            self.update_flag = True
            if log:
                self.logger.info(f"RECV: Player {payload} left the game.")

        # player joined/moved
        elif msg_type == 3:
            x, y = self.connection.receive_struct("ff")
            if payload not in self.players:
                self.logger.info(f"RECV: Player {payload} joined the game at ({round(x)},{round(y)})")
                if payload != self.id:
                    self.opponents.append(payload)
                new_player = Player(x, y, payload)

            else:
                self.logger.info(f"RECV: Player {payload} moved to ({round(x)},{round(y)})")
                if payload in self.ignored_opponents:
                    self.ignored_opponents.remove(payload)
                elif payload == self.id:
                    self.ignored_opponents.clear()
                new_player = self.players[payload]
                new_player.position[0] = x
                new_player.position[1] = y

            self.players[payload] = new_player
            self.update_flag = True

        # shot finished
        elif msg_type == 4:
            self.logger.error(
                "RECV: ERROR! WRONG BOT PROTOCOL VERSION DETECTED (VER < 8)! THIS MSG_TYPE SHOULD NOT BE RECEIVED!")
            exit(1)

        # shot begin
        elif msg_type == 5:
            _, _ = self.connection.receive_struct("dd")

        # shot end
        elif msg_type == 6:
            #  discard all shot data
            _, _, length = self.connection.receive_struct("ddI")
            for i in range(length):
                _ = self.connection.receive_struct("ff")

        # game mode, deprecated
        elif msg_type == 7:
            self.logger.error(
                "RECV: ERROR! WRONG BOT PROTOCOL VERSION DETECTED (VER <= 8)! THIS MSG_TYPE SHOULD NOT BE RECEIVED!")
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
            self.logger.info(f"RECV: Map data changed. {payload} planets received.")
            self.choose_name()
            self.update_flag = True

        # unknown MSG_TYPE
        else:
            self.logger.warning(f"RECV: Unexpected message_type: '{msg_type}'\n\t- data: '{payload}'")
            # try to discard all data appended to this message type
            self.connection.discard_all(.1)
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
            return
        # update simulation field if flag is set
        if self.update_flag:
            self.logger.info("Update flag set, updating field...")
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
        if not list(set(self.opponents).difference(set(self.ignored_opponents))):
            self.logger.info("No remaining viable opponents.")
