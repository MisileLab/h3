import { A } from "@solidjs/router";
import { Button } from "./ui/button";
import { revt } from "~/definition";
import { useColorMode } from "@kobalte/core";

export default function NavBar() {
  const {setColorMode, colorMode} = useColorMode();
  return (
    <div class="flex justify-end flex-row bg-background w-screen sticky h-14 align-middle border-primary border-b-2">
      <div class="my-auto mr-auto flex flex-row">
        <A href="/" class="ml-2">
          <img src="/logo.svg" alt="icon" width={40} height={40} />
        </A>
        <h1 class="my-auto font-bold text-3xl ml-4">AnA</h1>
      </div>
      <div class="my-auto ml-auto flex flex-row">
        <div onClick={()=>{setColorMode(revt(colorMode()))}} class="justify-center items-center flex"><img src={`/${colorMode()}/mode.svg`} class="text-primary mr-2" alt={colorMode()} width={30}/></div>
        <Button variant="secondary"><A href="/about">About</A></Button>
        <Button variant="secondary" class="ml-2">Projects</Button>
        <Button class="ml-2 mr-2">신청</Button>
      </div>
    </div>
  );
}
