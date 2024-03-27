import React, { useState } from 'react';

function ModelSelector({ setModel }) {
    const [selectedFile, setSelectedFile] = useState(null);

    const handleFileChange = (event) => {
        setSelectedFile(event.target.files[0].name);
    };

    const handleSubmit = async () => {
        // TODO set url from env
        const response = await fetch(process.env.REACT_APP_STREAM_POSE_ML_API_ENDPOINT + '/set_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            body: JSON.stringify({ filename: selectedFile }),
        });

        if (response.ok) {
            const jsonResponse = await response.json();
            setModel(jsonResponse.result);
        } else {
            console.error('Setting the model failed');
        }
    };

    return (
        <div>
            <h2>Usage Instructions</h2>
            <p>Refer to project documentation for instructions on how to save your trained classifier as a "pickle" file. Place this file within "data/trained_models" at the root of the application (you may need to create these folders). Then select the model here:</p>
            <input className="left" type="file" onChange={handleFileChange} />
            <br />
            <p>Set the application to use the model below. If successful, a video component will appear as well as the option to select a Bluetooth device.</p>
            <button className="left" onClick={handleSubmit} disabled={!selectedFile}>
                Set Model
            </button>
        </div>
    );
}

export default ModelSelector;
