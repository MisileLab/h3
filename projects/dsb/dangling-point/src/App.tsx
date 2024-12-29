import { For, JSX } from "solid-js";

interface Machine {
  name: string;
  color: string;
  position: Point[];
  inputs: Point[];
  outputs: Point[];
}

interface Point {
  x: number;
  y: number;
}

function sheet(content: JSX.Element = <></>, addistyle: string = "opacity-30 border-solid border-r-2 border-b-2 border-gray") {
  return <div class={`w-full h-full flex flex-col ${addistyle}`}>
    {content}
  </div>;
}

function App() {
  return (
    <div class="bg-white w-screen h-screen flex flex-row">
      <For each={Array(10)}>
        {() => sheet((
            <For each={Array(10)}>  
              {() => sheet(<></>, "border-solid border-b-2 border-gray")}
            </For>
          ), "opacity-30 border-solid border-r-2 border-gray")
        }
      </For>
    </div>
  );
}

export default App;
