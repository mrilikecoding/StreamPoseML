import './App.css';
// import Api from "./helpers/api"
import VideoStream from './VideoStream';
import ModelSelector from './ModelSelector';  
import React, { useState } from 'react';

function App() {
  const [isVideoStreamOn, setVideoStream] = useState(true);
  const toggleVideoStream = () => {
    setVideoStream(!isVideoStreamOn);
  }

  return (
    <div className="App">
      <h1>AI Tango</h1>
      <div className="container">
        <div className='column'>
          <ModelSelector />
          <hr />
          <hr />
        </div>
        <div className='column'>
          <button onClick={toggleVideoStream}>
            {isVideoStreamOn ? 'Turn off keypoint classification' : 'Turn on keypoint classification'} 
          </button>
          <hr />
          <VideoStream isOn={isVideoStreamOn} />
        </div>
      </div>
      </div>
  );
}

export default App;
