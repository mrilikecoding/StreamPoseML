# import bluetooth
import os


class BluetoothDevice:
    def __init__(self, port=1):
        # Env set in docker-compose.yml
        self.addr = os.getenv("BLUETOOTH_DEVICE_MAC")
        self.uuid = os.getenv("SPP_UUID")
        self.port = port
        # self.sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        # self.sock.connect((self.addr, port))

    def send(self, message):
        # self.sock.send(message)
        print("message sent")

    def receive(self, buffer_size=1024):
        # return self.sock.recv(buffer_size)
        return "fake message"

    def close(self):
        # self.sock.close()
        print("close socket")
