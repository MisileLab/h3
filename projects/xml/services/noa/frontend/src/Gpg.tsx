import { Title } from "@solidjs/meta";
import { For, createEffect, createSignal } from "solid-js";
import { backendurl, host, schale_url } from "./config";
import { getCookie, setCookie } from "./Upload";

export interface KeyStore {
  pubkey: string;
  name: string;
}

export default function Gpg() {
  const [gpgs, setgpgs] = createSignal<KeyStore[]>([]);
  createEffect(async ()=>{
    if (getCookie("jwt") === null) {
      window.location.href = `${schale_url}?redirect=${host}/gpg`;
    }
    if (new URLSearchParams(window.location.search).get("jwt") !== null) {
      setCookie("jwt", new URLSearchParams(window.location.search).get("jwt"), 21);
    }
    let r = await fetch(`${backendurl}`, {
      headers: {
        jwt: getCookie('jwt')
      }
    })
    setgpgs(await r.text() as unknown as KeyStore[])
  })
  return (
    <div class="w-screen h-screen bg-ctp-crust flex justify-center items-center">
      <Title>{window.location.pathname}</Title>
      <select class="w-fit h-fit px-2 bg-ctp-overlay0">
        <For each={gpgs()}>
          {(i,_)=><option value={i.pubkey}>{i.name}</option>}
        </For>
      </select>
    </div>
  );
};

