import React, { useEffect, useRef, useState } from 'react';

import io from "socket.io-client";
import {
    PoseLandmarker,
    FilesetResolver,
    // DrawingUtils
} from "@mediapipe/tasks-vision";

let poseLandmarker = undefined;
const createPoseLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
        },
        runningMode: "VIDEO"

    });
};

function VideoLoad() {
    return (
        <div>
            <h2>Load Video test</h2>
        </div>
    )
}

export default VideoLoad;
