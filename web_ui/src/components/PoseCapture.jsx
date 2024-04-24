import React, { useState } from 'react';


function PoseCapture({ handleVideoToggle, videoStreamer, videoLoader }) {
    const [isVideoStreamOn, setVideoStreamOn] = useState(false);

    const toggleVideoStream = () => {
        handleVideoToggle(!isVideoStreamOn);
        setVideoStreamOn(!isVideoStreamOn)
    }

    return (
        <div className='content-center p-4'>
            <button className='btn btn-primary w-full' onClick={toggleVideoStream}>
                {isVideoStreamOn ? 'Stop streaming' : 'Classify from webcam stream'}
            </button>
            {isVideoStreamOn ? videoStreamer : null}
        </div>
    )
}

export default PoseCapture;