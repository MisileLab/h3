import { createFileUploader } from "@solid-primitives/upload";
import axios from "axios";
import { createSignal, createEffect } from "solid-js";
import { getCookie, setCookie, getUrls } from "../config";

export default function uploadButton() {
  const {files, selectFiles} = createFileUploader();
  const [path, setPath] = createSignal("");
  let backendurl: string;
  createEffect(()=>{
    const {schale_url, host, backendurl: _backendurl} = getUrls();
    backendurl = _backendurl;
    if (getCookie("jwt") === null) {
      window.location.href = `${schale_url}?redirect=${host}/noa/f/upload`;
    }
    if (new URLSearchParams(window.location.search).get("jwt") !== null) {
      setCookie("jwt", new URLSearchParams(window.location.search).get("jwt") || "", 21);
    }
  })
  return (
    <>
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
    </>
  );
}