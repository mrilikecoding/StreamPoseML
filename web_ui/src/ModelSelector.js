import React, { useState } from 'react';

function ModelSelector({ setModel }) {
    const [selectedFile, setSelectedFile] = useState(null);

    const handleFileChange = (event) => {
        setSelectedFile(event.target.files[0].name);
    };

    const handleSubmit = async () => {
        // TODO set url from env
        const response = await fetch(process.env.REACT_APP_POSE_PARSER_API_ENDPOINT + '/set_model', {
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
            <h3>(Work in progress)</h3>
            <p>This application's Model Builder toolkit will export a model in the format appropriate to load up here. See notebooks for examples. However, any Python model (sci-kit, XGBoost, etc) can be loaded into the Poser Client. However, for now, an appropriate transformer must be written and specified within the application that maps the incoming keypoint data to the columns expected by the model.</p>
            <p><i>This will be made easier with the addition of a "model schema" that should accompany a selected model that should replace the need for specific transformers to be written.</i></p>
            <p>To use your model here, create a dictionary with the key "classifier" and set the value to the trained model.</p>
            <p>Within this same dictionary, include a key "model_data", where the value is a dictionary containing the key X_test. X_test should be a pandas dataframe of test data. This is only used to grab the expected columns for the model.</p>
            <p>Save this dictionary to a pickle file.</p>
            <p>Place your pickle with the trained model/data in the folder "data/trained_models", then select the model here:</p>
            <input className="left" type="file" onChange={handleFileChange} />
            <br />
            <button className="left" onClick={handleSubmit} disabled={!selectedFile}>
                Set Model
            </button>
        </div>
    );
}

export default ModelSelector;
