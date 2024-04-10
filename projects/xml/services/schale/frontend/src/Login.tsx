import { A } from "@solidjs/router";
import { JSX, createSignal } from "solid-js";
import statusCheck, { backendurl } from "./config";
import { Title } from "@solidjs/meta";

export default function Login(): JSX.Element {
  const [id, setID] = createSignal("");
  const [pw, setPW] = createSignal("");
  return (
    <div class="bg-ctp-crust w-screen h-screen flex flex-col justify-center items-center">
      <Title>schale/login</Title>
      <div class="bg-ctp-mantle w-fit h-fit flex flex-col py-12">
        <div class="w-full h-full gap-8 flex px-4 md:px-16 justify-center flex-col text-ctp-text">
          <h1 class="font-bold text-2xl md:text-3xl">Manage all with <span class="text-transparent bg-clip-text bg-gradient-to-br from-ctp-sky to-ctp-mauve">one account</span></h1>
          <input type="text" placeholder="id" class="bg-ctp-overlay0 placeholder:text-ctp-text pl-1" value={id()} onChange={(a)=>setID(a.target.value)} />
          <input type="password" placeholder="password" class="bg-ctp-overlay0 placeholder:text-ctp-text pl-1" value={pw()} onChange={(a)=>setPW(a.target.value)}/>
          <button class="bg-ctp-overlay0 w-full" onclick={async ()=>{
            let r = await fetch(`${backendurl}/login`, {
              headers: {
                userid: id(),
                pw: pw()
              },
              method: "POST"
            });
            if (r.status === 401) {
             alert('password or username failed');
             return;
            }
            if (statusCheck(r)) return;
            let t = await r.clone().text()
            t = t.slice(1,t.length-1)
            console.log(t);
            location.href = `https://${new URL(document.location.toString()).searchParams.get("redirect")}?jwt=${t}`;
          }}>Login</button>
          <A href="/register" class="w-full text-ctp-blue">No account? Click & Register</A>
        </div>
      </div>
    </div>
  );
};

