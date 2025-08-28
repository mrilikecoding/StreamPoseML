import threading
import time

from app import (  # type: ignore[import-not-found, attr-defined]
    app,
    check_connection_health,
    socketio,
)


def periodic_health_check():
    """Run health check every 30 seconds"""
    while True:
        time.sleep(30)
        try:
            check_connection_health()
        except Exception as e:
            print(f"[ERROR] Health check failed: {e}")


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

    # Start periodic health check in background thread
    health_thread = threading.Thread(target=periodic_health_check, daemon=True)
    health_thread.start()
    print("[INFO] Started periodic health check thread (runs every 30s)")

    socketio.run(app, host="0.0.0.0", port=5001, debug=True, use_reloader=True)
