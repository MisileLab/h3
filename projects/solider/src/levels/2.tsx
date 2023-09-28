import { Navigate } from "@solidjs/router";
import { createEffect, createSignal } from "solid-js";
import solution from '../styles/solution.module.css';

export function Level2() {
    let [pw, setpw] = createSignal("");
    let realpw = "A_CSS_THING";

    createEffect(() => {
        if (pw() == realpw) {
            return (<Navigate href="/3"></Navigate>);
        }
    });

    return (
        <div>
            <h1 class={solution.level2}>Level 2</h1>
            <input type="text" placeholder="i use arch btw" value={pw()} onChange={(e) => setpw(e.target.value)}></input>
        </div>
    );
}
