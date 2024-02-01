import { ColorMode } from "@kobalte/core";
import moment from "moment";
import { Moment } from "moment";

export function revt(t: ColorMode): ColorMode {
    if (t == "dark"){return "light"}else{return "dark"}
}

export function subtractDate(date: number | Moment, date2: number | Moment): Moment {
    if (typeof date !== 'number') {date = date.unix()}
    if (typeof date2 !== 'number') {date2 = date2.unix()}
    return moment.unix(date2 - date);
}
