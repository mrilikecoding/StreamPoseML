import React, { useState, useEffect } from "react";

const ConnectionHealth = ({ socket, connectionMetrics }) => {
  const [connectionStatus, setConnectionStatus] = useState("disconnected");
  const [lastHeartbeat, setLastHeartbeat] = useState(Date.now());
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const [showDetails, setShowDetails] = useState(false);
  const [hasReceivedFirstResponse, setHasReceivedFirstResponse] = useState(false);
  const [errorLog, setErrorLog] = useState([]);

  useEffect(() => {
    if (!socket) return;

    // Set up heartbeat interval
    const heartbeatInterval = setInterval(() => {
      socket.emit("ping_heartbeat");
      
      // Don't mark as degraded/critical during initialization
      if (hasReceivedFirstResponse) {
        // Check if we haven't received a pong in 15 seconds
        if (Date.now() - lastHeartbeat > 15000) {
          setConnectionStatus("degraded");
        }
        if (Date.now() - lastHeartbeat > 30000) {
          setConnectionStatus("critical");
        }
      }
    }, 5000);

    // Socket event handlers
    const handleConnect = () => {
      console.log("[ConnectionHealth] Socket connected");
      // Always start in initializing state on new connection
      setConnectionStatus("initializing");
      setReconnectAttempts(0);
      setLastHeartbeat(Date.now());
      setHasReceivedFirstResponse(false); // Reset on new connection
    };

    const handleDisconnect = () => {
      console.log("[ConnectionHealth] Socket disconnected");
      setConnectionStatus("disconnected");
    };

    const handlePongHeartbeat = (data) => {
      setLastHeartbeat(Date.now());
      setHasReceivedFirstResponse(true);
      setConnectionStatus("healthy");
    };

    const handleForceReconnect = (data) => {
      console.log("[ConnectionHealth] Server requested reconnection:", data);
      setConnectionStatus("reconnecting");
      
      // Log the forced reconnection
      const errorEntry = {
        time: new Date().toLocaleTimeString(),
        type: "Server Force Reconnect",
        message: `Reason: ${data.reason || "Unknown"}, Count: ${data.count || 0}`
      };
      setErrorLog(prev => [...prev.slice(-9), errorEntry]);
      
      socket.disconnect();
      setTimeout(() => {
        socket.connect();
      }, 1000);
    };

    const handleConnectionError = (error) => {
      console.error("[ConnectionHealth] Connection error:", error);
      setConnectionStatus("error");
      setReconnectAttempts(prev => prev + 1);
      
      // Add to error log with timestamp
      const errorEntry = {
        time: new Date().toLocaleTimeString(),
        type: "Connection Error",
        message: error.message || error.toString()
      };
      setErrorLog(prev => [...prev.slice(-9), errorEntry]); // Keep last 10 errors
    };

    const handleConnectionCheck = (data) => {
      // Respond to server's connection check
      socket.emit("connection_check_response", { timestamp: Date.now() });
    };

    // Register event handlers
    socket.on("connect", handleConnect);
    socket.on("disconnect", handleDisconnect);
    socket.on("pong_heartbeat", handlePongHeartbeat);
    socket.on("force_reconnect", handleForceReconnect);
    socket.on("connect_error", handleConnectionError);
    socket.on("connection_check", handleConnectionCheck);

    // Cleanup
    return () => {
      clearInterval(heartbeatInterval);
      socket.off("connect", handleConnect);
      socket.off("disconnect", handleDisconnect);
      socket.off("pong_heartbeat", handlePongHeartbeat);
      socket.off("force_reconnect", handleForceReconnect);
      socket.off("connect_error", handleConnectionError);
      socket.off("connection_check", handleConnectionCheck);
    };
  }, [socket, lastHeartbeat]);

  // Update status based on metrics from frame_result
  useEffect(() => {
    if (connectionMetrics?.connection_health) {
      setHasReceivedFirstResponse(true);
      // Use server-reported health if available
      if (connectionMetrics.connection_health === "critical") {
        setConnectionStatus("critical");
        
        // Log critical issues
        if (connectionMetrics.emit_failures > 0) {
          const errorEntry = {
            time: new Date().toLocaleTimeString(),
            type: "Emit Failures",
            message: `Failed to send ${connectionMetrics.emit_failures} messages to server`
          };
          setErrorLog(prev => {
            // Only add if not duplicate
            if (prev.length === 0 || prev[prev.length - 1].message !== errorEntry.message) {
              return [...prev.slice(-9), errorEntry];
            }
            return prev;
          });
        }
      } else if (connectionMetrics.connection_health === "degraded") {
        setConnectionStatus("degraded");
      } else if (connectionMetrics.connection_health === "healthy") {
        setConnectionStatus("healthy");
      }
      
      // Log burst warnings
      if (connectionMetrics.burst_warning) {
        const errorEntry = {
          time: new Date().toLocaleTimeString(),
          type: "Burst Warning",
          message: "Multiple classifications triggered too quickly"
        };
        setErrorLog(prev => [...prev.slice(-9), errorEntry]);
      }
    }
  }, [connectionMetrics]);

  const getStatusColor = () => {
    switch (connectionStatus) {
      case "healthy":
        return "bg-green-500";
      case "initializing":
        return "bg-blue-500 animate-pulse";
      case "degraded":
        return "bg-yellow-500";
      case "critical":
      case "error":
        return "bg-red-500";
      case "reconnecting":
        return "bg-orange-500 animate-pulse";
      case "disconnected":
      default:
        return "bg-gray-500";
    }
  };

  const getStatusText = () => {
    switch (connectionStatus) {
      case "healthy":
        return "Model Server Connected";
      case "initializing":
        return "Model Server Initializing";
      case "degraded":
        return "Model Server Degraded";
      case "critical":
        return "Model Server Critical";
      case "error":
        return "Model Server Error";
      case "reconnecting":
        return "Reconnecting to Model Server...";
      case "disconnected":
      default:
        return "Model Server Disconnected";
    }
  };

  const handleReconnect = () => {
    console.log("[ConnectionHealth] Manual reconnection triggered");
    setConnectionStatus("reconnecting");
    socket.disconnect();
    setTimeout(() => {
      socket.connect();
    }, 500);
  };

  const timeSinceHeartbeat = Math.round((Date.now() - lastHeartbeat) / 1000);

  return (
    <div className="card card-compact bg-base-100 p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="indicator">
            <span className={`indicator-item badge badge-xs ${getStatusColor()} animate-pulse`}></span>
            <div className={`w-4 h-4 rounded-full ${getStatusColor()}`}></div>
          </div>
          <div>
            <h3 className="font-semibold">{getStatusText()}</h3>
            {!socket ? (
              <p className="text-xs text-gray-500">
                Start streaming to connect
              </p>
            ) : connectionStatus === "disconnected" ? (
              <p className="text-xs text-gray-500">
                Waiting for stream to start...
              </p>
            ) : connectionStatus === "initializing" ? (
              <p className="text-xs text-gray-500">
                Waiting for first classification...
              </p>
            ) : (
              <p className="text-xs text-gray-500">
                Last heartbeat: {timeSinceHeartbeat}s ago
              </p>
            )}
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          {socket && connectionStatus === "critical" && timeSinceHeartbeat > 30 && (
            <span className="text-xs text-warning">
              Connection issues detected
            </span>
          )}
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="btn btn-sm btn-ghost"
          >
            {showDetails ? "Hide" : "Details"}
          </button>
        </div>
      </div>

      {showDetails && (
        <div className="mt-4 p-3 bg-base-200 rounded-lg">
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>
              <span className="font-semibold">Status:</span> {connectionStatus}
            </div>
            <div>
              <span className="font-semibold">Heartbeat:</span> {timeSinceHeartbeat}s ago
            </div>
            {reconnectAttempts > 0 && (
              <div>
                <span className="font-semibold">Reconnect Attempts:</span> {reconnectAttempts}
              </div>
            )}
            {connectionMetrics && (
              <>
                <div>
                  <span className="font-semibold">Emit Failures:</span> {connectionMetrics.emit_failures || 0}
                </div>
                <div>
                  <span className="font-semibold">Connection Issues:</span> {connectionMetrics.connection_issues || 0}
                </div>
                <div>
                  <span className="font-semibold">Last Emit:</span> {Math.round((connectionMetrics.time_since_last_emit_ms || 0) / 1000)}s ago
                </div>
                {connectionMetrics.burst_warning && (
                  <div className="col-span-2 text-warning">
                    <span className="font-semibold">⚠️ Burst Warning Active</span>
                  </div>
                )}
              </>
            )}
          </div>
          
          {/* Error Log */}
          {errorLog.length > 0 && (
            <div className="mt-3 pt-3 border-t border-base-300">
              <h4 className="font-semibold text-sm mb-2">Recent Issues:</h4>
              <div className="space-y-1 max-h-32 overflow-y-auto">
                {errorLog.map((error, index) => (
                  <div key={index} className="text-xs p-1 bg-base-100 rounded">
                    <span className="text-gray-500">{error.time}</span>
                    <span className="ml-2 font-semibold">{error.type}:</span>
                    <span className="ml-1">{error.message}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ConnectionHealth;