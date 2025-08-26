#!/usr/bin/env python3
"""
Test script for WebSocket recovery implementation.
Simulates connection issues to verify recovery mechanisms work.
"""

import json
import time

import socketio  # type: ignore[import-untyped]

# Create a Socket.IO client
sio = socketio.Client()

# Track connection state
connection_state = {"connected": False, "health_checks": 0, "reconnects": 0}


@sio.event
def connect():
    print("[TEST] Connected to server")
    connection_state["connected"] = True


@sio.event
def disconnect():
    print("[TEST] Disconnected from server")
    connection_state["connected"] = False
    connection_state["reconnects"] += 1


@sio.on("frame_result")
def on_frame_result(data):
    """Monitor frame results for connection health metrics"""
    if "connection_health" in data:
        print(f"[METRICS] Connection health: {data['connection_health']}")
    if "emit_failures" in data:
        print(f"[METRICS] Emit failures: {data['emit_failures']}")
    if "burst_warning" in data and data["burst_warning"]:
        print("[WARNING] Burst detected!")


@sio.on("force_reconnect")
def on_force_reconnect(data):
    """Handle server's force reconnect request"""
    print(f"[RECOVERY] Server requested reconnection: {data}")
    sio.disconnect()
    time.sleep(1)
    sio.connect("http://localhost:5001")


@sio.on("pong_heartbeat")
def on_pong_heartbeat(data):
    """Track heartbeat responses"""
    connection_state["health_checks"] += 1
    print(f"[HEARTBEAT] Pong received (total: {connection_state['health_checks']})")


def send_test_keypoints():
    """Send test keypoint data to trigger classification"""
    test_keypoints = json.dumps(
        {
            "keypoints": [[0.5, 0.5, 0.9] for _ in range(33)],  # Dummy keypoints
            "timestamp": time.time(),
        }
    )

    try:
        sio.emit("keypoints", test_keypoints)
        print("[TEST] Sent keypoints")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to send keypoints: {e}")
        return False


def test_normal_operation():
    """Test normal operation with heartbeats"""
    print("\n=== Testing Normal Operation ===")

    # Send heartbeats
    for i in range(3):
        sio.emit("ping_heartbeat")
        print(f"[TEST] Sent heartbeat {i + 1}")
        time.sleep(2)

    # Send some keypoints
    for i in range(5):
        if send_test_keypoints():
            print(f"[TEST] Keypoints batch {i + 1} sent")
        time.sleep(1)


def test_connection_degradation():
    """Simulate connection issues"""
    print("\n=== Testing Connection Degradation ===")

    # Disconnect to simulate connection loss
    print("[TEST] Simulating connection loss...")
    sio.disconnect()
    time.sleep(5)

    # Reconnect
    print("[TEST] Attempting reconnection...")
    try:
        sio.connect("http://localhost:5001")
        print("[TEST] Reconnected successfully")
    except Exception as e:
        print(f"[ERROR] Reconnection failed: {e}")


def main():
    """Run WebSocket recovery tests"""
    print("WebSocket Recovery Test Suite")
    print("=" * 40)

    # Connect to server
    try:
        print("[TEST] Connecting to server at localhost:5001...")
        sio.connect("http://localhost:5001")
    except Exception as e:
        print(f"[ERROR] Could not connect to server: {e}")
        print("Make sure the API server is running (make run-api)")
        return

    # Wait for connection
    time.sleep(2)

    if not connection_state["connected"]:
        print("[ERROR] Failed to establish connection")
        return

    # Run tests
    test_normal_operation()
    test_connection_degradation()
    test_normal_operation()  # Test recovery

    # Summary
    print("\n" + "=" * 40)
    print("Test Summary:")
    print(f"  Health checks received: {connection_state['health_checks']}")
    print(f"  Reconnection attempts: {connection_state['reconnects']}")
    status = "Connected" if connection_state["connected"] else "Disconnected"
    print(f"  Final connection state: {status}")

    # Cleanup
    sio.disconnect()
    print("\n[TEST] Test completed")


if __name__ == "__main__":
    main()
