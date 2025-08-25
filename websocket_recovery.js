// WebSocket Recovery Implementation for StreamPoseML Frontend
// Add this to your frontend code where you initialize the Socket.IO connection

// Initialize connection tracking variables
let connectionHealth = 'unknown';
let lastHeartbeat = Date.now();
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 5;

// Initialize Socket.IO with recovery options
const socket = io('your-backend-url', {
    reconnection: true,
    reconnectionAttempts: MAX_RECONNECT_ATTEMPTS,
    reconnectionDelay: 1000,
    reconnectionDelayMax: 5000,
    timeout: 20000,
    transports: ['websocket', 'polling'] // Prefer WebSocket
});

// Connection event handlers
socket.on('connect', () => {
    console.log('[INFO] WebSocket connected successfully');
    connectionHealth = 'healthy';
    reconnectAttempts = 0;
    
    // Start heartbeat
    startHeartbeat();
});

socket.on('disconnect', (reason) => {
    console.log('[WARNING] WebSocket disconnected:', reason);
    connectionHealth = 'disconnected';
    
    // Stop heartbeat during disconnect
    stopHeartbeat();
});

socket.on('connect_error', (error) => {
    console.error('[ERROR] Connection error:', error.message);
    connectionHealth = 'error';
    reconnectAttempts++;
    
    if (reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
        console.error('[CRITICAL] Max reconnection attempts reached');
        // Could trigger UI alert here
    }
});

// Recovery signal handlers
socket.on('force_reconnect', (data) => {
    console.log('[INFO] Server requested reconnection:', data);
    connectionHealth = 'reconnecting';
    
    // Force reconnection
    socket.disconnect();
    setTimeout(() => {
        socket.connect();
    }, 1000);
});

socket.on('connection_error', (data) => {
    console.error('[ERROR] Server reported connection error:', data);
    // Update UI to show connection issues
});

socket.on('connection_check', (data) => {
    console.log('[INFO] Server connection check:', data);
    // Respond to let server know we're alive
    socket.emit('connection_check_response', { timestamp: Date.now() });
});

// Heartbeat mechanism
let heartbeatInterval = null;

function startHeartbeat() {
    // Clear any existing interval
    stopHeartbeat();
    
    // Send heartbeat every 5 seconds
    heartbeatInterval = setInterval(() => {
        socket.emit('ping_heartbeat');
        
        // Check if we haven't received a pong in 15 seconds
        if (Date.now() - lastHeartbeat > 15000) {
            console.warn('[WARNING] No heartbeat response in 15 seconds');
            connectionHealth = 'degraded';
            
            // Try to recover
            if (Date.now() - lastHeartbeat > 30000) {
                console.error('[ERROR] No heartbeat in 30 seconds, reconnecting');
                socket.disconnect();
                socket.connect();
            }
        }
    }, 5000);
}

function stopHeartbeat() {
    if (heartbeatInterval) {
        clearInterval(heartbeatInterval);
        heartbeatInterval = null;
    }
}

// Handle heartbeat response
socket.on('pong_heartbeat', (data) => {
    lastHeartbeat = Date.now();
    connectionHealth = 'healthy';
});

// Monitor connection health in frame results
socket.on('frame_result', (data) => {
    // Update connection health from server metrics
    if (data.connection_health) {
        connectionHealth = data.connection_health;
        
        // Show warning if connection degraded
        if (data.connection_health === 'degraded' || data.connection_health === 'critical') {
            console.warn('[WARNING] Connection health:', data.connection_health);
            // Update UI to show connection warning
        }
    }
    
    // Check for burst warning
    if (data.burst_warning) {
        console.warn('[WARNING] Burst detected - multiple classifications too quickly');
    }
    
    // Process classification result
    if (data.classification !== undefined) {
        // Your existing classification handling code
        handleClassification(data);
    }
});

// Export connection health for UI
function getConnectionHealth() {
    return {
        status: connectionHealth,
        lastHeartbeat: lastHeartbeat,
        reconnectAttempts: reconnectAttempts,
        timeSinceHeartbeat: Date.now() - lastHeartbeat
    };
}

// Manual recovery function (can be called from UI)
function forceReconnect() {
    console.log('[INFO] Manual reconnection triggered');
    socket.disconnect();
    setTimeout(() => {
        socket.connect();
    }, 500);
}

// Example: Update UI with connection status
setInterval(() => {
    const health = getConnectionHealth();
    
    // Update connection indicator (example)
    const indicator = document.getElementById('connection-indicator');
    if (indicator) {
        indicator.className = `connection-${health.status}`;
        indicator.title = `Connection: ${health.status} (${Math.round(health.timeSinceHeartbeat/1000)}s ago)`;
    }
    
    // Show alert if connection is critical
    if (health.status === 'critical' || health.timeSinceHeartbeat > 30000) {
        // Show reconnection button or alert
        const alert = document.getElementById('connection-alert');
        if (alert) {
            alert.style.display = 'block';
            alert.textContent = 'Connection issues detected. Click to reconnect.';
            alert.onclick = forceReconnect;
        }
    }
}, 1000);