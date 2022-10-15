import socket
import struct
import sys


class SocketManager:
    def __init__(self, ip, port, version, recv_timeout):
        # set up socket and connect
        self.connected = False
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ip = ip
        self.port = port
        self.bot_ver = version
        self.recv_timeout = recv_timeout
        self.initialize()

    def initialize(self):
        self.connect()
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
                    print("Connection dropped unexpectedly during RECV.")
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
            print("Connection dropped unexpectedly during SEND.")
            exit(1)

    def receive_struct(self, struct_format):
        byte_struct = self.receive_bytes(struct.calcsize(struct_format))
        if not byte_struct:
            return None
        return struct.unpack(struct_format, byte_struct)

    def close(self):
        sys.stdout.write("Closing socket connection...")
        try:
            self.socket.close()
            print("success.")
        except TimeoutError:
            print("failed.\nUnable to drop the connection, maybe the connection is already dead?")
        self.connected = False

    def connect(self) -> bool:
        sys.stdout.write("Opening socket connection...")
        try:
            self.socket.connect((self.ip, self.port))
            self.connected = True
            print("success.")
            return True
        except (ConnectionRefusedError, ConnectionAbortedError):
            print(
                f"failed.\nUnable to establish connection, is server at {self.ip}:{self.port} really up and reachable?")
            exit(1)
