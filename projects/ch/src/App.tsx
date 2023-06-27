import "./App.css";

function App() {
  return (
    <div class="container">
      <h1>
        <select id="selectenc">
          <option value="enc">Encrypt</option>
          <option value="dec">Decrypt</option>
        </select> text with Rust
      </h1>
      <label for="bruhenc">Enctext</label>
      <div><textarea id="orgtext" name="orgtext" rows="10" cols="50"></textarea></div>
    </div>
  );
}

export default App;
