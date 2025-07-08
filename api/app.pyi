from flask import Flask
from flask_socketio import SocketIO

class App(Flask):
    socketio: SocketIO

app: App
socketio: SocketIO
