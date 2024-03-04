import { type Component } from 'solid-js';
import { ColorModeProvider, ColorModeScript, useColorMode} from '@kobalte/core';
import NavBar from './components/navbar';

const App: Component = () => {
  return (
    <div>
      <ColorModeScript />
      <ColorModeProvider>
        <div class="h-screen flex flex-col">
          <NavBar />
          <div class="flex flex-grow justify-center">
            <div class="flex flex-col justify-center mb-20 items-center">
              <img src={`/${(()=>{const {colorMode} = useColorMode();return colorMode();})()}/logo.svg`} alt="Lambda" width={200} />
              <h1 class="mt-4 font-bold text-5xl">Lambda</h1>
              <h2 class="mt-2 font-normal text-4xl">The simplest form</h2>
            </div>
          </div>
        </div>
      </ColorModeProvider>
    </div>
  );
};

export default App;
