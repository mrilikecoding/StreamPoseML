import './App.css';
// import Api from "./helpers/api"
import VideoStream from './VideoStream';

function App() {
  // const api = new Api();
  // const processVideo = () => {
  //   api
  //     .processVideo()
  //     .then((response) => console.log(response))
  //     .catch((err) => console.log(err));
  // };


  return (
    <div className="App">
      <header className="App-header">
        <VideoStream />
      </header>
    </div>
  );
}

export default App;
