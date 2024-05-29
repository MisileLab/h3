import { nullValue, NaNValue } from "./errors";

export function nullVerify<T>(v: T | null): T {
  if (v == null || v == undefined) {throw new nullValue();} else {return v;}
}

export function getItem(key: string): string { return nullVerify(localStorage.getItem(key)) }
export function getItems(key: Array<string>): object {
  const vl = new Array();
  for (const i of key) {
    vl.push(getItem(i));
  }
  return vl;
}

export function redirect(url: string) { location.href = `${location.href}${url}`; }
export function redirectExternal(url: string) { location.href = url; }

export function query(selector: string): HTMLElement { return nullVerify(document.querySelector(selector)) }
export function queryAll(selector: string): NodeListOf<HTMLElement> { return nullVerify(document.querySelectorAll(selector))  }
export function queryId(id: string): HTMLElement { return nullVerify(document.getElementById(id)) }

export type enumType = Record<string, any>;
export function keyToEnumv<K>(key: K, T: enumType): typeof T {
  return nullVerify(T[key as unknown as keyof typeof T])
}
export function enumvToKey<V>(value: V, T: enumType): string {
  return nullVerify(Object.keys(T)[Object.values(T).indexOf(value as unknown as typeof T)])
}

export function parseInt(value: string): number {
  const v = Number.parseInt(value);
  if (isNaN(v)) {throw NaNValue;} else {return v;}
}
