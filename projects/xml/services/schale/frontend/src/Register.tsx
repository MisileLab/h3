import { JSX } from "solid-js";

export default function Register(): JSX.Element {
  return (
    <div class="bg-ctp-crust w-screen h-screen flex flex-col justify-center items-center">
      <div class="bg-ctp-mantle w-fit h-fit flex flex-col py-12">
        <div class="w-full h-full gap-8 flex px-4 md:px-16 justify-center flex-col text-ctp-text">
          <h1 class="font-bold text-2xl md:text-3xl">Without tracker, <span class="text-transparent bg-clip-text bg-gradient-to-br from-ctp-sky to-ctp-mauve">with privacy</span></h1>
          <input type="text" placeholder="id" class="bg-ctp-overlay0 placeholder:text-ctp-text pl-1"/>
          <input type="password" placeholder="password" class="bg-ctp-overlay0 placeholder:text-ctp-text pl-1"/>
          <button class="bg-ctp-overlay0 w-full">Register</button>
        </div>
      </div>
    </div>
  );
};

