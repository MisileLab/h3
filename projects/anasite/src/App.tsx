import { type Component } from 'solid-js';
import { ColorModeProvider, ColorModeScript} from '@kobalte/core';
import NavBar from './components/navbar';

const App: Component = () => {
  return (
    <div>
      <ColorModeScript />
      <ColorModeProvider>
        <div class="h-screen flex flex-col">
          <NavBar/>
          <div class="flex flex-grow justify-center">
            <div class="flex flex-col justify-center mb-20 items-center">
              <img src="/logo.svg" alt="AnA" width={200} />
              <h1 class="mt-4 font-bold text-5xl">AnA</h1>
              <h1 class="mt-4 font-medium text-xl sm:text-2xl md:text-3xl">Application and Architecture</h1>
            </div>
          </div>
        </div>
      </ColorModeProvider>
    </div>
  );
};

export default App;
