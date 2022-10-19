import logging
import socket
import struct
import sys
import time

from spinners import Spinners


class SocketManager:
    def __init__(self, ip, port, retry_interval, version, recv_timeout):
        self.logger = logging.getLogger(__name__)

        # set up socket and establish connection
        self.connected = False
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ip = ip
        self.port = port
        self.retry_interval = retry_interval
        self.bot_ver = version
        self.recv_timeout = recv_timeout

    def initialize(self):
        self.establish_connection()
        self.discard_all(1)
        self.send_str(f"b {self.bot_ver}")

    # receives and discards all packages until it times out
    def discard_all(self, discard_timeout=1):
        timeout = self.socket.gettimeout()
        self.socket.settimeout(discard_timeout)
        while True:
            try:
                self.socket.recv(4096)
            except TimeoutError:
                break
        self.socket.settimeout(timeout)

    def receive_bytes(self, byte_count):
        timeout = self.socket.gettimeout()
        self.socket.settimeout(self.recv_timeout)

        buf = b''
        while byte_count:
            try:
                new_buf = self.socket.recv(byte_count)
                if not new_buf:
                    self.logger.error("Connection dropped unexpectedly during RECV.")
                    exit(1)

                buf += new_buf
                byte_count -= len(new_buf)
            except TimeoutError:
                buf = None
                break

        self.socket.settimeout(timeout)
        return buf

    def send_str(self, string):
        # trim whitespaces and add newline
        string = f"{string.strip()}\n"

        # calculate payload size
        payload = bytes(string, 'UTF-8')
        byte_count = len(payload)
        byte_i = 0
        try:
            while byte_i < byte_count:
                new_bytes = self.socket.send(payload[byte_i:])
                if not new_bytes:
                    raise BrokenPipeError

                byte_i += new_bytes
            return True
        except BrokenPipeError:
            self.logger.error("Connection dropped unexpectedly during SEND.")
            exit(1)

    def receive_struct(self, struct_format):
        byte_struct = self.receive_bytes(struct.calcsize(struct_format))
        if not byte_struct:
            return None
        return struct.unpack(struct_format, byte_struct)

    def close(self):
        self.logger.info("Closing socket connection...")
        try:
            self.socket.close()
            self.logger.info("success.")
        except TimeoutError:
            self.logger.warning("failed.\nUnable to drop the connection, maybe the connection is already dead?")
        self.connected = False

    def establish_connection(self):
        self.logger.info("Opening socket connection...")
        last_try = time.time()
        spinner = Spinners['bouncingBall'].value
        spinner_active = False
        index = 0
        while True:
            try:
                self.connect()
                self.connected = True
                if spinner_active:
                    sys.stdout.write("success.\n")
                self.logger.info("Socket connection established.")
                return
            except (ConnectionRefusedError, ConnectionAbortedError):
                try:
                    while time.time() - last_try <= self.retry_interval:
                        spinner_active = True
                        sys.stdout.write(
                            f"\r{spinner['frames'][index]} Waiting for connection on {self.ip}:{self.port}...")
                        index = (index + 1) % len(spinner['frames'][index])
                        time.sleep(spinner['interval'] / 1000)
                    last_try = time.time()
                except KeyboardInterrupt:
                    sys.stdout.write("\b\bkeyboard interrupt, exiting.\n")
                    exit(0)

    def connect(self):
        try:
            self.socket.connect((self.ip, self.port))
        except Exception as e:
            raise e
