import { open, Command } from '@tauri-apps/api/shell';
import { writeFile, readTextFile, exists } from '@tauri-apps/api/fs';
import "./App.css";
import './index.css';
import { createSignal } from 'solid-js';
import { invoke } from '@tauri-apps/api/tauri';

const [getState, modifyState] = createSignal("not growing now");

const sleep = (ms: number) => {
  return new Promise(resolve=>{
    setTimeout(resolve, ms)
  })
}

let repository = ""
let enabled: boolean = false;

function set_repository() {
  repository = prompt("Input Git Repository", "git@github.com/username/repository.git")!!;
}

async function githubSSH() {
  await open("https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent")
}

async function timerHandler() {
  if (!exists) {
    writeFile("grassgrower.txt", "MisileLaboratory | 1 >>= grass grower")
  }
  console.log("GrassGrower start")
  let org = await readTextFile("grassgrower.txt")
  let mod = org
  while (org == mod) {
    mod = `MisileLaboratory | ${invoke("random")} >>= grass grower`
    writeFile("grassgrower.txt", mod)
  }
  new Command("cd", ["repository"]).spawn()
  new Command("run-git-add", ["add", "-A"]).spawn()
  new Command("run-git-commit", ["commit", "-m", "auto commit"]).spawn()
  new Command("run-git-push", ["push"]).spawn()
  new Command("cd", [".."]).spawn()
  console.log("GrassGrower end")
}

async function buttonHandle() {
  console.log(enabled);
  enabled = !enabled
  console.log(enabled);
  if (enabled) {
    try {
      if (!exists("repository")) {
        new Command("run-git-clone", ["clone", repository, "repository"]).spawn()
      }
    } catch(e) {
      console.log(e)
      return;
    }
    modifyState("grass growing...")
    while (enabled) {
      timerHandler()
      sleep(86400000)
    }
  } else {
    modifyState("not growing now")
  }
  writeFile("enabled.txt", String(enabled))
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
      <button onClick={buttonHandle}>
      <div class="run-wrapper">
        <div class="run">
          <div class="growing-grass-wrapper">
          <b class="growing-grass">{getState()}</b>
          </div>
        </div>
      </div>
      </button>
      <button onClick={githubSSH}><div class="gen-ssh-key">
        <div class="daco-68893-1-parent">
          <img
            class="daco-68893-1-icon"
            alt=""
            src="/daco-68893-1@2x.png"
          />

          <b class="generate-ssh-key" style="font-family: SUIT;font-size: 24px; margin: none;">Generate ssh Key</b>
        </div>
        <b class="success">success!</b>
        <img
          class="arrow-forward-ios-icon"
          alt=""
          src="/arrow-forward-ios.svg"
        />
      </div></button>
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
