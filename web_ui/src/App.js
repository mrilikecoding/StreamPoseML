import './App.css';
// import Api from "./helpers/api"
import VideoStream from './VideoStream';
import ModelSelector from './ModelSelector';
import React, { useState, useEffect } from 'react';

function App() {
    const [isVideoStreamOn, setVideoStream] = useState(false);
    const [model, setModel] = useState(null);

    const toggleVideoStream = () => {
        setVideoStream(!isVideoStreamOn);
    }
    useEffect(() => {
        console.log(model);
    }, [model]);

    return (
        <div className="App">
            <h1>Stream Pose ML: Web UI</h1>
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
                            <VideoStream isOn={isVideoStreamOn} />
                        </div>
                    }
                    {!model && <h2>Select a model to get started</h2>}
                </div>
            </div>
        </div>
    );
}

export default App;
