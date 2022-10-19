from SocketManager import SocketManager
import appleBot
import cursor
import logging


# CONFIG
IP = "127.0.0.1"
PORT = 3490
RETRY_INTERVAL = 1
BOT_VERSION = 9
RECV_TIMEOUT = 0.1

logging.basicConfig(format='[%(asctime)s] [%(levelname)-8s] --- [%(module)-14s]: %(message)s',
                    level=logging.INFO,
                    force=True)

if __name__ == "__main__":
    cursor.hide()

    # initialize connection and wait for it to establish
    sock_manager = SocketManager(IP, PORT, RETRY_INTERVAL, BOT_VERSION, RECV_TIMEOUT)

    # initialize bot object
    bot = appleBot.AppleBot(sock_manager)

    # loop until connection breaks
    while sock_manager.connected:
        bot.loop()
