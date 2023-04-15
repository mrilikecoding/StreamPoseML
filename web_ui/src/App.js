import './App.css';
import Api from "./helpers/api"
import VideoStream from './VideoStream';

function App() {
  const api = new Api();
  const processVideo = () => {
    api
      .processVideo()
      .then((response) => console.log(response))
      .catch((err) => console.log(err));
  };

  const apiUrl = JSON.stringify(process.env);

  return (
    <div className="App">
      <header className="App-header">
        <VideoStream />
        <p>
          API: {apiUrl}
        </p>
        <button onClick={processVideo}>
          Process Videos
        </button>
      </header>
    </div>
  );
}

export default App;
