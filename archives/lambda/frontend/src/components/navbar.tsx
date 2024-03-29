import { A } from "@solidjs/router";
import { revt } from "~/definition";
import { useColorMode } from "@kobalte/core";
import { Button } from "./ui/button";

export default function NavBar() {
  const {setColorMode, colorMode} = useColorMode();
  return (
    <div class="flex justify-end flex-row bg-background w-full sticky h-14 align-middle border-primary border-b-2">
      <div class="my-2 mr-auto flex flex-row">
        <A href="/" class="ml-2">
          <img src={`${colorMode()}/logo.svg`} alt="icon" class="mt-0.5" width={43} />
        </A>
      </div>
      <div class="my-2 ml-auto flex flex-row">
        <A href="/ana" class="mr-2">
          <img src="/universal/ana.svg" alt="ana" class="mt-0.5" width={43} />
        </A>
        <A href="/admin" class="mr-2">
          <Button>Admin</Button>
        </A>
        <div onClick={()=>{setColorMode(revt(colorMode()))}} class="justify-center items-center flex"><img src={`/${colorMode()}/mode.svg`} class="text-primary mr-2" alt={colorMode()} width={30}/></div>
      </div>
    </div>
  );
}
