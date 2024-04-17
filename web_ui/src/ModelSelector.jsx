import React, { useState } from 'react';

function ModelSelector({ setModel }) {
    const [selectedFile, setSelectedFile] = useState(null);

    const handleFileChange = (event) => {
        setSelectedFile(event.target.files[0]);
    };

    const handleSubmit = async () => {
        const data = new FormData();
        data.append('filename', selectedFile.name);
        data.append('file', selectedFile);

        fetch(import.meta.env.VITE_STREAM_POSE_ML_API_ENDPOINT + '/set_model', {
            body: data,
            method: 'POST',
        })
            .then((res) => res.json())
            .then((data) => {
                setModel(data.result)
            })
            .catch((err) => console.error(err))
    };

    return (
        <div>
            <h2>Usage Instructions</h2>
            <p>Refer to project documentation for instructions on how to save your trained classifier as a "pickle" file. Place this file within "data/trained_models" at the root of the application (you may need to create these folders). Then select the model here:</p>
            <input type="file" onChange={handleFileChange} />
            <br />
            <p>Set the application to use the model below. If successful, a video component will appear as well as the option to select a Bluetooth device.</p>
            <div className='column'>
                <button onClick={handleSubmit} disabled={!selectedFile}>
                    Set Model
                </button>
            </div>
            <div className='column'>
                <div>{selectedFile ? selectedFile.name : "No model selected"}</div>
            </div>
        </div>
    );
}

export default ModelSelector;
