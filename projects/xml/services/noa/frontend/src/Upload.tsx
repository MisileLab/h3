import { createFileUploader } from "@solid-primitives/upload";
import { Title } from "@solidjs/meta";
import axios from "axios";
import { createEffect, createSignal } from "solid-js";
import { host, schale_url } from "./config";

export function setCookie(name: string, value: string, days: number) {
    var expires = "";
    if (days) {
      var date = new Date();
      date.setTime(date.getTime() + (days*24*60*60*1000));
      expires = "; expires=" + date.toUTCString();
    }
    document.cookie = name + "=" + (value || "")  + expires + "; path=/";
}
export function getCookie(name: string) {
    var nameEQ = name + "=";
    var ca = document.cookie.split(';');
    for(var i=0;i < ca.length;i++) {
      var c = ca[i];
      while (c.charAt(0)==' ') c = c.substring(1,c.length);
      if (c.indexOf(nameEQ) == 0) return c.substring(nameEQ.length,c.length);
    }
    return null;
}

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
      <div class="w-fit h-fit p-6 bg-ctp-overlay0 flex flex-row">
        <span class="text-ctp-subtext0 text-4xl px-2">/</span>
        <input class="w-fit h-fit p-2 bg-ctp-overlay1 text-ctp-subtext1" placeholder="path" onchange={(a)=>setPath(a.target.value)} value={path()}/>
        <span class="text-ctp-subtext0 text-4xl px-2">/</span>
        <button onclick={()=>{
          selectFiles(async ([{source,name,size,file}]) => {
            console.log(source,name,size,file);
            if (files()[0] !== undefined) {
              for (const i of files()) {
                console.log(i.name);
                const fd = new FormData();
                fd.append("file", i.file);
                await axios.post(`${schale_url}/uploadfile`, fd, {
                  onUploadProgress: (r) => {console.log(r);},
                  headers: {
                    jwt: getCookie("jwt")
                  }
                })
              }
            }
          });
        }} class="bg-ctp-overlay1 p-2 text-ctp-text">Select File</button>
      </div>
    </div>
  );
};

