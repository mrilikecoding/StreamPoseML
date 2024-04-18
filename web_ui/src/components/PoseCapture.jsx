import React, { useState } from 'react';


function PoseCapture({ handleVideoToggle, videoStreamer, videoLoader }) {
    const [isVideoStreamOn, setVideoStreamOn] = useState(false);

    const toggleVideoStream = () => {
        handleVideoToggle(!isVideoStreamOn);
        setVideoStreamOn(!isVideoStreamOn)
    }

    return (
        <div>
            <div>
                <button className='btn btn-primary' onClick={toggleVideoStream}>
                    {isVideoStreamOn ? 'Stop streaming' : 'Classify from webcam stream'}
                </button>
                {isVideoStreamOn ? videoStreamer : null}
            </div>
        </div>
    )
}

export default PoseCapture;