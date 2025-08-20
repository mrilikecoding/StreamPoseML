from app import app, socketio  # type: ignore[import-not-found, attr-defined]

if __name__ == "__main__":
    print(
        """
        Now running...

        ==========================
        --------------------------
        ._._ StreamPoseML:API _._.
        --------------------------
        ==========================

        """
    )
    socketio.run(app, host="0.0.0.0", port=5001, debug=True, use_reloader=True)
