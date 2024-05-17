import { createFileUploader } from "@solid-primitives/upload";
import { Title } from "@solidjs/meta";
import axios from "axios";
import { createEffect, createSignal } from "solid-js";
import { backendurl, getCookie, host, schale_url, setCookie } from "./config";

export default function Upload() {
  const {files, selectFiles} = createFileUploader();
  const [path, setPath] = createSignal("");
  createEffect(()=>{
    if (getCookie("jwt") === null) {
      window.location.href = `${schale_url}?redirect=${host}/upload`;
    }
    if (new URLSearchParams(window.location.search).get("jwt") !== null) {
      setCookie("jwt", new URLSearchParams(window.location.search).get("jwt"), 21);
    }
  })
  return (
    <div class="w-screen h-screen bg-ctp-crust flex justify-center items-center">
      <Title>{window.location.pathname}</Title>
      <div class="w-fit h-fit p-6 flex flex-row">
        <span class="text-ctp-subtext0 text-4xl px-2">/</span>
        <input class="w-fit h-fit p-2 border-ctp-overlay0 border-solid border-2 bg-ctp-crust text-ctp-subtext1" placeholder="path" onchange={(a)=>setPath(a.target.value)} value={path()}/>
        <span class="text-ctp-subtext0 text-4xl px-2">/</span>
        <button onclick={()=>{
          selectFiles(async ([{source,name,size,file}]) => {
            console.log(source,name,size,file);
            if (files()[0] !== undefined) {
              for (const i of files()) {
                console.log(i.name);
                const fd = new FormData();
                fd.append("file", i.file);
                await axios.post(`${backendurl}/uploadfile`, fd, {
                  onUploadProgress: (r) => {console.log(r);},
                  headers: {
                    jwt: getCookie("jwt"),
                    path: path()
                  }
                })
              }
            }
          });
        }} class="border-ctp-overlay0 border-solid border-2 p-2 text-ctp-text">Select File</button>
      </div>
    </div>
  );
};

