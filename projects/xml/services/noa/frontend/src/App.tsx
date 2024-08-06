import { Title } from "@solidjs/meta";
import { useLocation, useParams } from "@solidjs/router";
import { VsFolder, VsFile } from "solid-icons/vs";
import { For, JSX, createEffect, createMemo, createSignal } from "solid-js";
import statusCheck, { backendurl, host } from "./config";

export interface File {
  name: string,
  size: number,
  dir: boolean
}

export function Icon(props: {isDirectory: boolean}): JSX.Element {
  return props.isDirectory ? <VsFolder /> : <VsFile />
}

export function FileName(name: string, path: string, isDir: boolean) {
  if (path.startsWith("/")) {path=path.slice(1)}
  return (
    <a href={`${isDir?host:`${backendurl}/file`}/${path===""?"":`${path}/`}${name}`} class="flex flex-row items-center gap-1">
      <Icon isDirectory={isDir} />
      <span class="text-ctp-sky">{name}</span>
    </a>
  );
}

export default function App() {
  const params = useParams();
  const pathname = createMemo(()=>useLocation().pathname.slice("/noa/f".length))
  let [res, setRes] = createSignal<File[]>([]);
  createEffect(async ()=>{
    let r = await fetch(`${backendurl}/files`, {
      headers: {
        "path": params.path.slice("/noa/f".length)
      }
    });
    if (statusCheck(r)) {return;}
    setRes(JSON.parse(await r.text()) as unknown as File[]);
  });
  return (
    <div class="w-screen h-screen bg-ctp-crust flex justify-center items-center">
      <Title>{pathname()}</Title>
      <div class="border-ctp-overlay0 border-solid border-2 w-fit h-fit flex flex-row gap-2 p-4 text-ctp-text">
        <div class="flex flex-col grow gap-2">
          <p class="font-bold">Name</p>
          {pathname() != "/" && FileName("..", "..", true)}
          <For each={res()}>
            {(i,_) => FileName(i.name, params.path, i.dir)}
          </For>
        </div>
        <div class="flex flex-col gap-2">
          <p class="font-bold">Size (Bytes)</p>
          {pathname() != "/" && <p>dir</p>}
          <For each={res()}>
            {(i,_) => <p>{!i.dir ? i.size : "dir"}</p>}
          </For>
        </div>
      </div>
    </div>
  );
};

