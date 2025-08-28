import React, { useState } from 'react';


function PoseCapture({ handleVideoToggle, videoStreamer, videoLoader, connectionHealth }) {
    const [isVideoStreamOn, setVideoStreamOn] = useState(false);

    const toggleVideoStream = () => {
        handleVideoToggle(!isVideoStreamOn);
        setVideoStreamOn(!isVideoStreamOn)
    }

    return (
        <div className='content-center p-4 h-full flex flex-col'>
            {connectionHealth && (
                <div className='mb-4 flex-shrink-0'>
                    {connectionHealth}
                </div>
            )}
            <button className='btn btn-primary w-full mb-4 flex-shrink-0' onClick={toggleVideoStream}>
                {isVideoStreamOn ? 'Stop streaming' : 'Classify from webcam stream'}
            </button>
            {isVideoStreamOn && (
                <div className='flex-grow min-h-0'>
                    {videoStreamer}
                </div>
            )}
        </div>
    )
}

export default PoseCapture;