import React, { useEffect, useRef, useState } from "react";
import io from "socket.io-client";

function VideoStream() {
  const localVideoRef = useRef();
  const socketRef = useRef();
  const [results, setResults] = useState(null);

  useEffect(() => {
    // Initialize the WebRTC connection
    async function initWebRTC() {
        const localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        if (localVideoRef.current) {
            localVideoRef.current.srcObject = localStream;
        }

        // Get the frame rate of the connected camera
        const videoTrack = localStream.getVideoTracks()[0];
        const settings = videoTrack.getSettings();
        let frameRate = settings.frameRate;
        if (frameRate >= 30) {
            frameRate = 30;
        }
        // Calculate the interval between frames in milliseconds
        const frameInterval = 1000 / frameRate;

        // Set up an interval to send frames periodically
        const intervalId = setInterval(() => {
            sendFrame();
        }, frameInterval);


        // Clean up the interval when the component is unmounted
        return () => clearInterval(intervalId);
    }

    initWebRTC();

    // Set up the Socket.IO connection and event listeners
    socketRef.current = io.connect("http://localhost:5000");
    socketRef.current.on("frame_result", (data) => {
        setResults(data);
    });


    /**
     * Capture a frame from a video element and convert it to a base64 encoded string.
     *
     * @param {HTMLVideoElement} video - The video element to capture the frame from.
     * @returns {string} - The base64 encoded frame data.
     */
    function captureFrame(video) {
        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        return canvas.toDataURL("image/jpeg", 0.8);
    }

    /**
     * Send a captured frame to the server via a Socket.IO event.
     */
    function sendFrame() {
        const video = document.getElementById("localVideo");
        if (video) {
            const frameData = captureFrame(video);
            socketRef.current.emit("frame", frameData);
        }
    }

    // Clean up the Socket.IO connection when the component is unmounted
    return () => {
        socketRef.current.disconnect();
    };

  }, []);

  return (
    <div>
      <video id="localVideo" ref={localVideoRef} autoPlay muted></video>
      <h2>Results:</h2>
        {results ? <pre>{JSON.stringify(results, null, 2)}</pre> : <p>Awaiting server response...</p>}
    </div>
  );
}

export default VideoStream;
