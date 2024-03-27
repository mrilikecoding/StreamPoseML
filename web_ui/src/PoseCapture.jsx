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
                {videoLoader}
            </div>
            <div>
                <button className='button' onClick={toggleVideoStream}>
                    {isVideoStreamOn ? 'Turn off keypoint classification' : 'Turn on keypoint classification'}
                </button>
                {isVideoStreamOn ? videoStreamer : null}
            </div>
        </div>
    )
}

export default PoseCapture;