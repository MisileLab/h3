import { VsFolder, VsFile } from "solid-icons/vs";
import { For, type JSX, createEffect, createSignal } from "solid-js";
import { useStore } from "@nanostores/solid";
import { path } from "../stores"
import statusCheck, { getUrls } from "../config";

export interface File {
  name: string,
  size: number,
  dir: boolean
}

export function Icon(props: {isDirectory: boolean}): JSX.Element {
  return props.isDirectory ? <VsFolder /> : <VsFile />
}

// i use ai on this
function simplifyUrl(url: string) {
  // Step 1: Remove double slashes
  let simplified = url.replace(/\/\/+/g, '/');
  
  // Split the URL into segments
  const segments = simplified.split('/');
  
  // Initialize an array to hold the processed segments
  let processedSegments = [];
  
  // Process each segment
  for (let segment of segments) {
      if (segment === '..') {
          // Move one level up
          if (processedSegments.length > 0) {
              processedSegments.pop();
          }
      } else if (segment !== '.') {
          // Add the segment to the processed segments
          processedSegments.push(segment);
      }
  }
  
  // Join the processed segments back together
  return processedSegments.length == 0?"/":processedSegments.join('/');
}


export function FileName(name: string, path: string, isDir: boolean, host: string, backendurl: string): JSX.Element {
  if (path.startsWith("/")) {path=path.slice(1)}
  return (
    <div class="flex flex-row items-center gap-1 cursor-pointer" onClick={()=>{
      window.location.href = `${isDir?`${host}/noa/f`:`${backendurl}/file`}/${isDir?(`?path=${simplifyUrl(`${path}/${name}`)}`):(`${path===""?"":`${path}/`}${name}`)}`;}
    }>
      <Icon isDirectory={isDir} />
      <span class="text-ctp-sky">{name}</span>
    </div>
  );
}

export default function App() {
  const {backendurl, host} = getUrls();
  const pathname = useStore(path)
  let [res, setRes] = createSignal<File[]>([]);
  createEffect(async ()=>{
    let r = await fetch(`${backendurl}/files`, {
      headers: {
        "path": `.${pathname()}`
      }
    });
    if (statusCheck(r)) {return;}
    setRes(JSON.parse(await r.text()) as unknown as File[]);
  });
  return (
    <>
      <div class="flex flex-col grow gap-2">
        <p class="font-bold">Name</p>
        {pathname() != "/" && FileName("..", pathname(), true, host, backendurl)}
        <For each={res()}>
          {(i,_) => FileName(i.name, pathname(), i.dir, host, backendurl)}
        </For>
      </div>
      <div class="flex flex-col gap-2">
        <p class="font-bold">Size (Bytes)</p>
        {pathname() != "/" && <p>dir</p>}
        <For each={res()}>
          {(i,_) => <p>{!i.dir ? i.size : "dir"}</p>}
        </For>
      </div>
    </>
  );
};

