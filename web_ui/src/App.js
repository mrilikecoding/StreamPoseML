import logo from './logo.svg';
import './App.css';
import Api from "./helpers/api"

function App() {
  const api = new Api();
  const processVideo = () => {
    api
      .processVideo()
      .then((response) => console.log(response))
      .catch((err) => console.log(err));
  };

  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Hello World!
        </p>
        <button onClick={processVideo}>
          Process Videos
        </button>
      </header>
    </div>
  );
}

export default App;
