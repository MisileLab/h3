import { render } from "solid-js/web";
import { Router, Route, Routes } from "@solidjs/router";
import { Level1 } from "./levels/1";
import { Level2 } from "./levels/2";

render(
  () => (
    <Router>
      <Routes>
        <Route path="/1" component={Level1} />
        <Route path="/2" component={Level2} />
      </Routes>
    </Router>
  ),
  document.getElementById("root")!!
);