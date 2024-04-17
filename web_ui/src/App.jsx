import io from "socket.io-client";

import './App.css';
// import Api from "./helpers/api"
import VideoStream from './VideoStream';
import VideoLoad from './VideoLoad';
import PoseCapture from './PoseCapture';
import ModelSelector from './ModelSelector';
import WebBluetooth from './WebBluetooth';
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
        <div className="App">
            <h1>StreamPose ML Web Client</h1>
            <div>
                <div>
                    <div className='column'>
                        <ModelSelector setModel={setModel} />
                    </div>
                    {model ? model : "Select model to begin classification"}
                    {results ? <pre>{JSON.stringify(results, null, 2)}</pre> : <p>Awaiting server response...</p>}
                </div>
                {/* <WebBluetooth 
                    deviceServiceUUID={DEVICE_SERVICE_UUID}
                    deviceCharacteristicUUID={DEVICE_CHARACTERISTIC_UUID}
                    classifierResult={classifierResult}
                /> */}
            </div>
            <div className="container">
                <PoseCapture 
                    handleVideoToggle={handleVideoToggle}
                    videoLoader={<VideoLoad />}
                    videoStreamer={<VideoStream 
                        handlePoseResults={handlePoseResults} 
                    />}
                />
            </div>
        </div>
    );
}

export default App;
