import { createSignal } from "solid-js";
import { invoke } from "@tauri-apps/api/tauri";
import "./App.css";
import './index.css';

let repository = "";

function logging() {
  console.log("asdsad");
  return undefined;
}

function set_repository() {
  repository = prompt("Input Git Repository", "https://github.com/username/repository")!!;
}

function App() {
  return (
    <div class="frame-parent">
      <div class="ua5fr1nbyxqstosvqxinmbvzdwrfsg-wrapper">
        <img
          class="ua5fr1nbyxqstosvqxinmbvzdwrfsg-icon"
          alt=""
          src="/0ua5fr1nbyxqstosvqxinmbvzdwrfsgucgczplda0enheepn-oxqcwtkyiyg2j1rpwj159exwalk399nmnsq-31@2x.png"
        />
      </div>
      <button onClick={logging}>
      <div class="run-wrapper">
        <div class="run">
          <div class="growing-grass-wrapper">
          <b class="growing-grass">Growing Grass.......</b>
          </div>
        </div>
      </div>
      </button>
      <a href="https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent"><div class="gen-ssh-key">
        <div class="daco-68893-1-parent">
          <img
            class="daco-68893-1-icon"
            alt=""
            src="/daco-68893-1@2x.png"
          />

          <b class="generate-ssh-key">Generate ssh Key</b>
        </div>
        <b class="success">success!</b>
        <img
          class="arrow-forward-ios-icon"
          alt=""
          src="/arrow-forward-ios.svg"
        />
      </div></a>
      <div class="real-log">
        <b class="logs">Logs</b>
      </div>
      <div class="git-repository-input">
        <div class="daco-68893-1-parent">
          <img
            class="daco-68893-1-icon"
            alt=""
            src="/daco-68893-1@2x.png"
          />

          <button onClick={set_repository}><b class="generate-ssh-key" style="font-family: SUIT;font-size: 24px; margin: none;">Input Git Repository</b></button>
        </div>
        <b class="success1">success!</b>
        <img
          class="arrow-forward-ios-icon"
          alt=""
          src="/arrow-forward-ios.svg"
        />
      </div>
      <img class="image-5-icon" alt="" src="./image-5@2x.png" />

      <b class="your-grass-status">Your Grass Status</b>
      <b class="current-status">Current Status</b>
      <b class="auto-grass-grower">Auto Grass Grower</b>
      <img
        class="ua5fr1nbyxqstosvqxinmbvzdwrfsg-icon1"
        alt=""
        src="/0ua5fr1nbyxqstosvqxinmbvzdwrfsgucgczplda0enheepn-oxqcwtkyiyg2j1rpwj159exwalk399nmnsq-31@2x.png"
      />

      <img
        class="ua5fr1nbyxqstosvqxinmbvzdwrfsg-icon2"
        alt=""
        src="/0ua5fr1nbyxqstosvqxinmbvzdwrfsgucgczplda0enheepn-oxqcwtkyiyg2j1rpwj159exwalk399nmnsq-31@2x.png"
      />

      <div class="wrapper">
        <b class="b">샌즈와 대화 하는법</b>
      </div>
      <div class="container">
        <b class="b">샌즈와 대화 하는법</b>
      </div>
      <div class="frame">
        <b class="b">샌즈와 대화 하는법</b>
      </div>
      <img class="frame-child" alt="" src="/frame-7.svg" />

      <img class="frame-item" alt="" src="/frame-13.svg" />

      <img class="frame-inner" alt="" src="/frame-9.svg" />
    </div>
  )
};

export default App;
