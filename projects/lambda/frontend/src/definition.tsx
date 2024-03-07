import { ColorMode } from "@kobalte/core";
import { GraphQLClient } from "graphql-request";
import { Setter, Accessor } from "solid-js";

export let url: string;
console.log(import.meta.env.DEV)
if (import.meta.env.DEV) {
    url = "http://127.0.0.1:8000"
} else {
    url = "https://lmbackend.misile.xyz"
}

export const client = new GraphQLClient(`${url}/graphql`)

export function revt(t: ColorMode): ColorMode {
    if (t == "dark"){return "light"}else{return "dark"}
}

export type signal<T> = [Accessor<T>, Setter<T>];

export interface User {
    name: string,
    pnumber: string,
    me: string,
    why: string,
    time: number,
    portfolio: string
}

export function setCookie(cookie_name: string, value: string, days: number) {
    var exdate = new Date();
    exdate.setDate(exdate.getDate() + days);
    var cookie_value = encodeURI(value) + ((days == null) ? '' : ';`expires=' + exdate.toUTCString());
    document.cookie = cookie_name + '=' + cookie_value;
}

export function getCookie(cookie_name: string) {
    var x: string, y: string;
    var val = document.cookie.split(';');
    for (var i = 0; i < val.length; i++) {
        x = val[i].substring(0, val[i].indexOf('='));
        y = val[i].substring(val[i].indexOf('=') + 1);
        x = x.replace(/^\s+|\s+$/g, '');
        if (x == cookie_name) {
            return decodeURI(y);
        }
    }
}

