import React, { useEffect, useRef, useState } from 'react';
import { PoseLandmarker, FilesetResolver, DrawingUtils } from "@mediapipe/tasks-vision";

import io from "socket.io-client";

const USE_GPU = true;
const MODEL_LITE_URI = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" 
const MODEL_FULL_URI = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task" 
const MODEL_HEAVY_URI = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task" 

const VideoStream = ({ handleClassification }) => {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const [posePresence, setPosePresence] = useState(null);
    const [keypointProcessingSpeed, setKeypointProcessingSpeed] = useState(null);

    const [results, setResults] = useState(null);
    const socketRef = useRef();
    

    
    useEffect(() => {
        // For drawing keypoints
        const canvasElement = canvasRef.current;
        const video = videoRef.current;
        const canvasCtx = canvasElement.getContext("2d");
        const drawingUtils = new DrawingUtils(canvasCtx);

        let poseLandmarker;
        let animationFrameId;

        const initializePoseDetection = async () => {
            // For configuration options see
            // https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/web_js
            try {
                const vision = await FilesetResolver.forVisionTasks(
                    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
                );
                poseLandmarker = await PoseLandmarker.createFromOptions(
                    vision, {
                    baseOptions: { 
                        modelAssetPath: MODEL_HEAVY_URI, 
                        delegate: USE_GPU ? "GPU" : "CPU",
                    },
                    numPoses: 1,
                    runningMode: "VIDEO",
                    // minPoseDetectionConfidence: 0.90,
                    // minPosePresenceConfidence: 0.90,
                    // minTrackingConfidence: 0.90,
                });
                detectPose();
            } catch (error) {
                console.log("Error initializing pose landmark detection.", error)
            }
        };

        const detectPose = () => {
            const FPS = 60 // cap max FPS @TODO make configurable 
            setTimeout(()=> {
                if (videoRef.current && videoRef.current.readyState >= 2) {
                    let startTimeMs = performance.now();
                    // const start = Date.now() // profile start
                    poseLandmarker.detectForVideo(videoRef.current, startTimeMs, (result) => {
                        canvasElement.width = video.clientWidth;
                        canvasElement.height = video.clientHeight;
                        canvasCtx.save();
                        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
                        for (const landmark of result.landmarks) {
                          drawingUtils.drawLandmarks(landmark, {
                            radius: (data) => {
                                if (data) {
                                    DrawingUtils.lerp(data.from.z, -0.15, 0.1, 5, 1)
                                }
                            }
                          });
                          drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);
                        }
                        canvasCtx.restore();
                    });
                    // setKeypointProcessingSpeed(Date.now() - start); // profile end
                }
                requestAnimationFrame(detectPose);
            }, 1000 / FPS)
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
        // function (video) {
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
            <div id="videoContainer">
                <video id="localVideo" ref={videoRef} autoPlay muted></video>
                <canvas id="videoStreamCanvas" ref={canvasRef} ></canvas>
                <pre>
                    {keypointProcessingSpeed}
                </pre>
            </div>
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
