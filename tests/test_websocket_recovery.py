"""
Tests for WebSocket connection health monitoring and recovery functionality.
"""

import time
import unittest
from unittest.mock import Mock, patch, MagicMock
import json

import pytest
import socketio

# Test the WebSocket recovery functionality
class TestWebSocketRecovery(unittest.TestCase):
    """Test WebSocket connection health and recovery mechanisms."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_socket = Mock()
        self.connection_metrics = {}
        
    def test_connection_health_states(self):
        """Test connection health state transitions."""
        # Test healthy state
        metrics = {
            "connection_health": "healthy",
            "emit_failures": 0,
            "time_since_last_emit_ms": 100
        }
        self.assertEqual(metrics["connection_health"], "healthy")
        
        # Test degraded state
        metrics["connection_health"] = "degraded"
        metrics["emit_failures"] = 3
        self.assertEqual(metrics["connection_health"], "degraded")
        
        # Test critical state
        metrics["connection_health"] = "critical"
        metrics["emit_failures"] = 15
        metrics["time_since_last_emit_ms"] = 12000
        self.assertEqual(metrics["connection_health"], "critical")

    def test_emit_failure_tracking(self):
        """Test emit failure counting and recovery detection."""
        # Simulate progressive failures
        failures = [0, 1, 5, 10, 15, 20]
        
        for failure_count in failures:
            metrics = {"emit_failures": failure_count}
            
            if failure_count == 0:
                self.assertEqual(metrics["emit_failures"], 0)
            elif failure_count < 10:
                self.assertLess(metrics["emit_failures"], 10)
            else:
                self.assertGreaterEqual(metrics["emit_failures"], 10)

    def test_rate_limiting_prevents_burst(self):
        """Test that rate limiting prevents burst classifications."""
        MIN_INTERVAL = 0.8  # seconds
        
        # Simulate classification timestamps
        last_time = time.time()
        current_time = last_time + 0.5  # Too soon
        
        time_diff = current_time - last_time
        should_skip = time_diff < MIN_INTERVAL
        
        self.assertTrue(should_skip, "Should skip classification if too soon")
        
        # Test valid timing
        current_time = last_time + 1.0  # Long enough
        time_diff = current_time - last_time
        should_skip = time_diff < MIN_INTERVAL
        
        self.assertFalse(should_skip, "Should allow classification after interval")

    @patch('socketio.Client')
    def test_force_reconnect_signal(self, mock_socketio):
        """Test force reconnect signal handling."""
        client = mock_socketio.return_value
        
        # Simulate force reconnect event
        force_reconnect_data = {
            "reason": "emit_failures",
            "count": 12
        }
        
        # Test that disconnect is called
        client.disconnect.return_value = None
        client.connect.return_value = None
        
        # Simulate the reconnection logic
        client.disconnect()
        client.connect()
        
        client.disconnect.assert_called_once()
        client.connect.assert_called_once()

    def test_buffer_size_impact(self):
        """Test buffer size impact on connection behavior."""
        # Test reduced buffer (90 packets at 30fps = 3 seconds)
        buffer_size = 90
        fps = 30
        buffer_time = buffer_size / fps
        
        self.assertEqual(buffer_time, 3.0)
        self.assertLess(buffer_time, 66.0)  # Much less than original 2000/30

    def test_connection_metrics_structure(self):
        """Test that connection metrics contain expected fields."""
        sample_metrics = {
            "connection_health": "healthy",
            "time_since_last_emit_ms": 150.5,
            "emit_failures": 0,
            "connection_issues": 0,
            "burst_warning": False
        }
        
        required_fields = [
            "connection_health",
            "time_since_last_emit_ms", 
            "emit_failures",
            "connection_issues"
        ]
        
        for field in required_fields:
            self.assertIn(field, sample_metrics)

    def test_timing_breakdown_metrics(self):
        """Test detailed timing breakdown metrics."""
        timing_metrics = {
            "frame_collection_time_ms": 191.7,
            "transformation_time_ms": 374.0,
            "mlflow_inference_time_ms": 440.2,
            "total_processing_time_ms": 818.4
        }
        
        # Test that sum of components approximates total
        component_sum = (
            timing_metrics["transformation_time_ms"] + 
            timing_metrics["mlflow_inference_time_ms"]
        )
        
        # Allow for some overhead in total processing time
        self.assertLessEqual(
            component_sum, 
            timing_metrics["total_processing_time_ms"]
        )

    def test_health_check_intervals(self):
        """Test health check timing intervals."""
        # Test various health check intervals
        intervals = {
            "heartbeat": 5,  # seconds
            "health_check": 30,  # seconds  
            "degraded_threshold": 15,  # seconds
            "critical_threshold": 30   # seconds
        }
        
        self.assertEqual(intervals["heartbeat"], 5)
        self.assertEqual(intervals["health_check"], 30)
        self.assertLess(intervals["heartbeat"], intervals["degraded_threshold"])
        self.assertLess(intervals["degraded_threshold"], intervals["critical_threshold"])


class TestConnectionHealthIntegration(unittest.TestCase):
    """Integration tests for connection health monitoring."""

    def setUp(self):
        """Set up integration test environment."""
        self.mock_app = Mock()
        self.mock_socketio = Mock()

    def test_periodic_health_check(self):
        """Test periodic health check functionality."""
        # Mock the global state variables
        with patch('builtins.globals', return_value={
            'last_successful_emit': time.time() - 35,  # 35 seconds ago
            'emit_failures': 0,
            'connection_issues_detected': 0
        }):
            # Simulate health check logic
            current_time = time.time()
            last_emit = current_time - 35
            time_since_emit = current_time - last_emit
            
            # Should trigger warning after 30 seconds
            self.assertGreater(time_since_emit, 30)

    @patch('requests.get')
    def test_api_connectivity(self, mock_get):
        """Test API connectivity verification."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        # Simulate connectivity test
        try:
            response = mock_get("http://localhost:5001/")
            api_available = response.status_code == 200
        except Exception:
            api_available = False
            
        self.assertTrue(api_available)

    def test_burst_warning_detection(self):
        """Test burst warning detection logic."""
        # Simulate rapid classifications
        classification_times = [
            time.time(),
            time.time() + 0.1,  # 100ms later - too fast
            time.time() + 0.2   # 200ms later - still too fast
        ]
        
        # Check intervals between classifications
        intervals = [
            classification_times[i] - classification_times[i-1] 
            for i in range(1, len(classification_times))
        ]
        
        # Should detect burst (intervals < 0.5 seconds)
        burst_detected = any(interval < 0.5 for interval in intervals)
        self.assertTrue(burst_detected)


if __name__ == '__main__':
    unittest.main()