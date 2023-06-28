import "./App.css";
import { invoke } from '@tauri-apps/api';

function App() {
  function encordec() {
    const selectenc = document.getElementById("selectenc") as HTMLInputElement;
    const orgtext = document.getElementById("orgtext") as HTMLInputElement;
    const restext = document.getElementById("restext") as HTMLInputElement;
    const salt = document.getElementById("salt") as HTMLInputElement;
    const enc = selectenc.value;
    const text = orgtext.value;
    const saltstr = salt.value;
    let a: string;
    if (enc == "enc") {
      a = "encrypt";
    } else {
      a = "decrypt";
    }
    invoke(a, { text, saltstr })
    .then((text) => {
      restext.value = text as string;
    })
    .catch((error) => {
      console.error(error);
    });
  }

  return (
    <div class="container">
      <h1>
        <select id="selectenc">
          <option value="enc">Encrypt</option>
          <option value="dec">Decrypt</option>
        </select> text with Rust
      </h1>
      <label for="orgtext">orgtext</label>
      <div><textarea id="orgtext" name="orgtext" rows="10" cols="50"></textarea></div>
      <label for="restext">restext</label>
      <div><textarea id="restext" name="restext" rows="10" cols="50"></textarea></div>
      <label for="salt">salt</label>
      <div><input id="salt" name="salt"></input></div>
      <button onclick={encordec}>start</button>
    </div>
  );
}

export default App;
