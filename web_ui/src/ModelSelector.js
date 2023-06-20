import React, { useState } from 'react';

function ModelSelector() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [result, setResult] = useState('');

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0].name);
  };

  const handleSubmit = async () => {
    // TODO set url from env
    const response = await fetch('http://localhost:5001/set_model', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*'
      },
      body: JSON.stringify({ filename: selectedFile }),
    });

    if (response.ok) {
      const jsonResponse = await response.json();
      setResult(jsonResponse.result);
    } else {
      console.error('Upload failed');
    }
  };

  return (
    <div>
      <h2>Usage Instructions</h2>
      <p>Any Python model (sci-kit, XGBoost, etc) can be loaded into the Poser Client. However, for now, an appropriate transformer must be written and specified within the application that maps the incoming keypoint data to the columns expected by the model. <i>This will be made easier with the addition of a schema that replaces the need for specific transformers to be written.</i></p>
      <p>To use your model here, create a dictionary with the key "classifier" and set the value to the trained model.</p>
      <p><i>Temporary: This step will be made easier.</i> But for now, within this same dictionary, include a key "model_data", where the value is a dictionary containing the key X_test. X_test should be a pandas dataframe of test data. This is only used to grab the expected columns for the model.</p>
      <p>Save this dictionary to a pickle file.</p>
      <p>Place your trained model in the folder "data/trained_models", then select the model here:</p>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleSubmit} disabled={!selectedFile}>
        Set Model
      </button>
      {selectedFile && <p>Selected model: {selectedFile}</p>}
      {result && <p>Server model status: {result}</p>}
    </div>
  );
}

export default ModelSelector;
