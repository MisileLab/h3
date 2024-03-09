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
  const exdate = new Date();
  exdate.setDate(exdate.getDate() + days);
  const cookie_value = encodeURI(value) + ((days == null) ? '' : ';`expires=' + exdate.toUTCString());
  document.cookie = cookie_name + '=' + cookie_value;
}

export function getCookie(cookie_name: string) {
  var x: string, y: string;
  const val = document.cookie.split(';');
  for (let i = 0; i < val.length; i++) {
    x = val[i].substring(0, val[i].indexOf('='));
    y = val[i].substring(val[i].indexOf('=') + 1);
    x = x.replace(/^\s+|\s+$/g, '');
    if (x == cookie_name) {
      return decodeURI(y);
    }
  }
}

export const endTime = 1710514800;

export function formatBytes(bytes: number, decimals: number = 2) {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];

  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

