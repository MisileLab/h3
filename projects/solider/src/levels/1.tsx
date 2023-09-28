import { Navigate } from "@solidjs/router";
import { createEffect, createSignal } from "solid-js";
import util from '../styles/util.module.css';

export function Level1() {
    let [pw, setpw] = createSignal("");
    let realpw = "0xc0ffee";

    createEffect(() => {
        if (pw() == realpw) {
            return (<Navigate href="/2"></Navigate>);
        }
    })

    return (
        <div>
            <h1>Level 1</h1>
            <input type="text" value={pw()} onInput={(e) => setpw(e.target.value)} />
            <p class={util.invisible}>Password is 0xc0ffee</p>
        </div>
    );
}
