import { JSX } from "solid-js";
import NavBar from "./components/navbar";
import { ColorModeProvider, ColorModeScript } from "@kobalte/core";

export default function AnA(): JSX.Element {
  return (
    <div>
      <ColorModeScript />
      <ColorModeProvider>
        <NavBar />
      </ColorModeProvider>
    </div>
  );
}
