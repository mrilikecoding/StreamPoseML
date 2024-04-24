import io from "socket.io-client";

import './App.css';
import VideoStream from './components/VideoStream';
import VideoLoad from './components/VideoLoad';
import PoseCapture from './components/PoseCapture';
import ModelSelector from './components/ModelSelector';
import WebBluetooth from './components/WebBluetooth';
import React, { useState, useEffect, useRef } from 'react';

function App() {
    const [isVideoStreamOn, setVideoStream] = useState(false);
    const [classifierResult, setClassifierResult] = useState(null);
    const [model, setModel] = useState(null);
    const [results, setResults] = useState(null);

    const socketRef = useRef();

    useEffect(() => {
        console.log("Classifier selected:", model);
    }, [model]);
    
    useEffect(() => {
        if (isVideoStreamOn) {
            // Set up the Socket.IO connection and event listeners
            socketRef.current = io.connect(import.meta.env.VITE_STREAM_POSE_ML_API_ENDPOINT);
            console.log("Socket Open");
            socketRef.current.on("frame_result", (data) => {
                setResults(data);
                handleClassification(data)
            });
        }
        
        return () => {
            if (socketRef.current !== undefined && socketRef.current !== null) {
                socketRef.current.disconnect();
                console.log("Socket Closed");
            }
        }
    }, [isVideoStreamOn])


    // Bluetooth - TODO abstract this out so that we can select what actuator to use
    const DEVICE_SERVICE_UUID = import.meta.env.VITE_BLUETOOTH_DEVICE_SERVICE_UUID.toLowerCase();
    const DEVICE_CHARACTERISTIC_UUID = import.meta.env.VITE_BLUETOOTH_DEVICE_CHARACTERISTIC_UUID.toLowerCase();

    const handleClassification = (result) => {
        setClassifierResult(result);
    };

    const handlePoseResults = (results) => {
        socketRef.current.emit("keypoints", results);
    }
    
    const handleVideoToggle = (value)  => {
        setVideoStream(value);
    }

    return (
        <div className="grid grid-cols-4 gap-4 p-6 content-stretch">
            <div className="prose col-span-full">
                <h1 className="">
                    StreamPose ML Web Client
                </h1>
            </div>
            <div className="prose col-span-2">
                <div role="alert" className={model ? "alert alert-success text-success-content" : "alert alert-warning text-warning-content" }>
                    <div className="col-span-2">
                        <ModelSelector setModel={setModel} />
                        <h3>
                            {model ? model : "No model selected."}
                        </h3>
                    </div>
                </div>
                <div className="col-span-2 card card-compact bg-base-100 px-4">
                    <h2>Classifier Result</h2>
                    {results ? <pre>{JSON.stringify(results, null, 2)}</pre> : <p>Not yet sending data.</p>}
                </div>
                <WebBluetooth classifierResult={classifierResult} />
            </div>
            <div className="col-span-2">
                <div className="fixed w-1/2">
                    <PoseCapture 
                        handleVideoToggle={handleVideoToggle}
                        videoLoader={<VideoLoad />}
                        videoStreamer={<VideoStream 
                            handlePoseResults={handlePoseResults} 
                            />}
                    />
                </div>
            </div>
        </div>
    );
}

export default App;
