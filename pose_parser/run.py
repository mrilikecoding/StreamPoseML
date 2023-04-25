from api.app import app, socketio

if __name__ == "__main__":
    print(
        """
        Now running...

        ╔═══════════╗
        ║ -.-.-.-.- ║
        ║ POSER:API ║
        ║ .-.-.-.-. ║
        ╚═══════════╝
        
        """
    )
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, use_reloader=False)
