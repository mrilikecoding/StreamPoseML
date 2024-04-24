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
        <>
            <h2>Select Trained Model</h2>
            <p>Refer to project documentation for instructions on how to save your trained classifier as a "pickle" file. Select the model here:</p>
            <div className='join'>
                <input className='file-input file-input-bordered w-full max-w-xs join-item' type="file" onChange={handleFileChange} />
                <button className='btn btn-primary join-item' onClick={handleSubmit} disabled={!selectedFile}>
                    Set Model
                </button>
            </div>

        </>
    );
}

export default ModelSelector;
