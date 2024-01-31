import { ColorMode } from "@kobalte/core";

export function revt(t: ColorMode): ColorMode {
    if (t == "dark"){return "light"}else{return "dark"}
}
