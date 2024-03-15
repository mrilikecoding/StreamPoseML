import './App.css';
// import Api from "./helpers/api"
import VideoStream from './VideoStream';
import VideoLoad from './VideoLoad';
import ModelSelector from './ModelSelector';
import WebBluetooth from './WebBluetooth';
import React, { useState, useEffect } from 'react';

function App() {
    const [isVideoStreamOn, setVideoStream] = useState(false);
    const [classifierResult, setClassifierResult] = useState(null);
    const [model, setModel] = useState(null);

    const toggleVideoStream = () => {
        setVideoStream(!isVideoStreamOn);
    }
    useEffect(() => {
        console.log(model);
    }, [model]);


    // Bluetooth - TODO abstract this out so that we can select what actuator to use
    const DEVICE_SERVICE_UUID = process.env.REACT_APP_BLUETOOTH_DEVICE_SERVICE_UUID.toLowerCase();
    const DEVICE_CHARACTERISTIC_UUID = process.env.REACT_APP_BLUETOOTH_DEVICE_CHARACTERISTIC_UUID.toLowerCase();

    const handleClassification = (result) => {
        setClassifierResult(result);
    };

    return (
        <div className="App">
            <h1>StreamPose ML Web Client</h1>
            <div className="container">
                <VideoLoad 
                />
            </div>
            <div>
                <WebBluetooth 
                    deviceServiceUUID={DEVICE_SERVICE_UUID}
                    deviceCharacteristicUUID={DEVICE_CHARACTERISTIC_UUID}
                    classifierResult={classifierResult}
                />
            </div>
            <div className="container">
                <div className='column'>
                    <ModelSelector setModel={setModel} />
                </div>
                <div className='column'>
                    {/* Render VideoStream only if a model is selected */}
                    {model &&
                        <div>
                            <span className="green">
                                {model}
                            </span>
                            <button className='button' onClick={toggleVideoStream}>
                                {isVideoStreamOn ? 'Turn off keypoint classification' : 'Turn on keypoint classification'}
                            </button>
                            <VideoStream 
                                handleClassification={handleClassification} 
                                isOn={isVideoStreamOn} 
                            />
                        </div>
                    }
                    {!model && <h2>Select a model to get started</h2>}
                </div>
            </div>
        </div>
    );
}

export default App;
