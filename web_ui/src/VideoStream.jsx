import React, { useEffect, useRef, useState } from 'react';
import { PoseLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import poseLandmarkerTask from "./models/pose_landmarker_lite.task"

import io from "socket.io-client";

const VideoStream = ({ handleClassification }) => {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const [posePresence, setPosePresence] = useState(null);

    const [results, setResults] = useState(null);
    const socketRef = useRef();
    
    useEffect(() => {
        let poseLandmarker;
        let animationFrameId;

        const initializePoseDetection = async () => {
            try {
                const vision = await FilesetResolver.forVisionTasks(
                    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
                );
                poseLandmarker = await PoseLandmarker.createFromOptions(
                    vision, {
                    baseOptions: { modelAssetPath: poseLandmarkerTask },
                    numPoses: 1,
                    runningMode: "VIDEO"
                });
                detectPose();
            } catch (error) {
                console.log("Error initializing pose landmark detection.", error)
            }
        };

        const drawLandmarks = (landmarksArray) => {
            const canvas = canvasRef.current;
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = 'white';
        
            landmarksArray.forEach(landmarks => {
                landmarks.forEach(landmark => {
                    const x = landmark.x * canvas.width;
                    const y = landmark.y * canvas.height;
        
                    ctx.beginPath();
                    ctx.arc(x, y, 5, 0, 2 * Math.PI); // Draw a circle for each landmark
                    ctx.fill();
                });
            });
        };

        const detectPose = () => {
            if (videoRef.current && videoRef.current.readyState >= 2) {
                const detections = poseLandmarker.detectForVideo(videoRef.current, performance.now());
                // setPosePresence(detections.handednesses.length > 0);

                // Assuming detections.landmarks is an array of landmark objects
                if (detections.landmarks) {
                    drawLandmarks(detections.landmarks);
                }
            }
            requestAnimationFrame(detectPose);
        }

        const startWebcam = async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                videoRef.current.srcObject = stream;
                await initializePoseDetection();
            } catch (error) {
                console.error("Error accessing webcam:", error);
            }
        };

        startWebcam();

        // // Initialize the WebRTC connection
        // async function initWebRTC() {
        //     const localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        //     if (videoRef.current) {
        //         videoRef.current.srcObject = localStream;
        //     }

        //     // Get the frame rate of the connected camera
        //     const videoTrack = localStream.getVideoTracks()[0];
        //     const settings = videoTrack.getSettings();
        //     let frameRate = settings.frameRate;
        //     if (frameRate >= 30) {
        //         frameRate = 30;
        //     }

        //     // TODO remove this - experimenting with frame rate
        //     // frameRate = 1
        //     // Calculate the interval between frames in milliseconds
        //     // const frameInterval = 1000 / frameRate;

        //     // TODO this is a rate limit based on processing time in the backend
        //     // if this interval (in ms) is too low, the backend will not be able to keep up
        //     // and the socket connection will hang and reset
        //     const frameInterval = 250;

        //     // Set up an interval to send frames periodically
        //     let intervalId;
        //     if (USE_CLIENTSIDE_POSE_ESTIMATION) {
        //         intervalId = setInterval(async () => {
        //             sendKeypoints();
        //         }, frameInterval);
        //     } else {
        //         intervalId = setInterval(() => {
        //             sendFrame();
        //         }, frameInterval);
        //     }


        //     // Clean up the interval when the component is unmounted
        //     return () => clearInterval(intervalId);
        // }


        // Set up the Socket.IO connection and event listeners
        // socketRef.current = io.connect(process.env.REACT_APP_STREAM_POSE_ML_API_ENDPOINT);
        // socketRef.current.on("frame_result", (data) => {
        //     setResults(data);
        //     handleClassification(data)
        // });


        /**
         * Capture a frame from a video element and convert it to a base64 encoded string.
         *
         * @param {HTMLVideoElement} video - The video element to capture the frame from.
         * @returns {string} - The base64 encoded frame data.
         */
        // function captureFrame(video) {
        //     const canvas = document.createElement("canvas");
        //     canvas.width = video.videoWidth;
        //     canvas.height = video.videoHeight;
        //     const ctx = canvas.getContext("2d");
        //     ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        //     return canvas.toDataURL("image/jpeg", 0.8);
        // }

        /**
         * Capture a frame from a video element and extract keypoints.
         *
         * @param {HTMLVideoElement} video - The video element to capture the frame from.
         */
        // let lastVideoTime = -1;
        // let currentResults = {};
        // function captureFrameKeypoints(video) {
        //     // Perform pose estimation on the captured frame
        //     const canvas = document.createElement("canvas");
        //     canvas.width = video.videoWidth;
        //     canvas.height = video.videoHeight;
        //     const ctx = canvas.getContext("2d");
        //     ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        //     if (canvas.height > 0 && canvas.width > 0) {
        //         let startTimeMs = performance.now();
        //         if (poseLandmarker && lastVideoTime !== video.currentTime) {
        //             lastVideoTime = video.currentTime;
        //             poseLandmarker.detectForVideo(video, startTimeMs, (results) => {
        //                 currentResults = results
        //             })

        //         } else {
        //             console.log("Poselandmarker not set")
        //         }
        //     }
        // }

        // /**
        //  * Send a captured frame to the server via a Socket.IO event.
        //  */
        // function sendFrame() {
        //     const video = document.getElementById("localVideo");
        //     if (video) {
        //         const frameData = captureFrame(video);
        //         socketRef.current.emit("frame", frameData);
        //     }
        // }

        /**
         * Send frame keypoints generated client-side to the server via a Socket.IO event.
         */
        // function sendKeypoints() {
        //     const video = document.getElementById("localVideo");
        //     if (video) {
        //         captureFrameKeypoints(video);
        //         socketRef.current.emit("keypoints", currentResults);
        //     }


        //     // Clean up the Socket.IO connection when the component is unmounted
        //     return () => {
        //         socketRef.current.disconnect();
        //     };

        // }
        return () => {
            // if (socketRef.current !== undefined && socketRef.current !== null) {
            //     socketRef.current.disconnect();
            // }
            if (videoRef.current && videoRef.current.srcObject) {
                videoRef.current.srcObject.getTracks().forEach(track => track.stop());
            }
            if (poseLandmarker) {
                poseLandmarker.close();
            }
            if (animationFrameId) {
                cancelAnimationFrame(animationFrameId);
            }
        }
    }, []);


    return (
        <div>
            <video  style={{ position: "absolute", width: "300px", height: "200px" }} id="localVideo" ref={videoRef} autoPlay muted></video>
            <canvas  style={{ position: "absolute", width: "300px", height: "200px" }} ref={canvasRef} ></canvas>
            <div className='container'>
                <div className='column'>
                    <h2>Debug:</h2>
                    {results ? <pre>{JSON.stringify(results, null, 2)}</pre> : <p>Awaiting server response...</p>}
                </div>
            </div>
        </div>
    );
}
export default VideoStream;
