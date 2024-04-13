import { JSX, createSignal } from "solid-js";
import statusCheck, { backendurl, host } from "./config";
import { Title } from "@solidjs/meta";

export default function Register(): JSX.Element {
  const [id, setid] = createSignal("");
  const [pw, setpw] = createSignal("");
  return (
    <div class="bg-ctp-crust w-screen h-screen flex flex-col justify-center items-center">
      <Title>schale/register</Title>
      <div class="bg-ctp-mantle w-fit h-fit flex flex-col py-12">
        <div class="w-full h-full gap-8 flex px-4 md:px-16 justify-center flex-col text-ctp-text">
          <h1 class="font-bold text-2xl md:text-3xl">Without tracker, <span class="text-transparent bg-clip-text bg-gradient-to-br from-ctp-sky to-ctp-mauve">with privacy</span></h1>
          <input type="text" placeholder="id" class="bg-ctp-overlay0 placeholder:text-ctp-text pl-1" value={id()} onchange={(a)=>setid(a.target.value)}/>
          <input type="password" placeholder="password" class="bg-ctp-overlay0 placeholder:text-ctp-text pl-1" value={pw()} onchange={(a)=>setpw(a.target.value)}/>
          <button class="bg-ctp-overlay0 w-full" onclick={async ()=>{
            let r = await fetch(`${backendurl}/check/${id()}`);
            if (statusCheck(r)) return;
            if (await r.text() === "true") {
              alert('name is in use')
              return;
            }
            r = await fetch(`${backendurl}/register`, {
              headers: {
                userid: id(),
                pw: pw()
              },
              method: "POST"
            });
            if (statusCheck(r)) return;
            alert(`${id()} registered`)
            window.location.href=`${host}/login?redirect=${new URL(document.location.toString()).searchParams.get("redirect")}`
          }}>Register</button>
        </div>
      </div>
    </div>
  );
};

