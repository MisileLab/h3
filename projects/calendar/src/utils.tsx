import { Event, EventDate, SimpleEvent } from './interfaces';

export function convertEventToHighlight(events: Event[]) {
  const highlights: Record<string, SimpleEvent[]> = {};
  events.forEach((e) => {
    if (e === undefined) {return;}
    let start = new Date(e.start.year, e.start.month - 1, e.start.day);
    const end = new Date(e.end.year, e.end.month - 1, e.end.day);
    while (start <= end) {
      const year = start.getFullYear();
      const month = start.getMonth() + 1;
      const day = start.getDate();
      if (highlights[`${year}:${month}:${day}`] === undefined) {
        highlights[`${year}:${month}:${day}`] = [];
      }
      highlights[`${year}:${month}:${day}`].push({
        year: year,
        month: month + 1,
        day: day,
        title: e.title,
        content: e.content,
        color: e.color,
        org: e
      });
      let newDate = start.setDate(start.getDate() + 1);
      start = new Date(newDate);
    }
  });
  return highlights;
}

export function handlingButton(d: Date, setDateState: Function, amount: number) {
  const _d = d;
  _d.setMonth(d.getMonth() + amount);
  setDateState([_d]);
}

export function getColor(cont: string) {
  if (cont == "일") {
    return "text-red-500";
  } else if (cont == "토") {
    return "text-blue-500";
  } else {
    return "text-black dark:text-ctp-subtext1";
  }
}

// null check function
export function nc(a: string | undefined) {return a === undefined ? "" : a }

export function convertDateToString(ios: EventDate, ioe: EventDate) {
  const a = (a: number) => {
    return a < 10 ? `0${a}` : a;
  };
  return {
    start: {
      full: `${ios.year}.${a(ios.month)}.${a(ios.day)} ${a(ios.hour)}:${a(
        ios.minute
      )}`,
      date: `${ios.year}-${a(ios.month)}-${a(ios.day)}`,
      time: `${a(ios.hour)}:${a(ios.minute)}`,
    },
    end: {
      full: `${ioe.year}.${a(ioe.month)}.${a(ioe.day)} ${a(ioe.hour)}:${a(
        ioe.minute
      )}`,
      date: `${ioe.year}-${a(ioe.month)}-${a(ioe.day)}`,
      time: `${a(ioe.hour)}:${a(ioe.minute)}`,
    },
  };
}

export function shallowEqual(object1: any, object2: any) {
  console.log(object1, object2);
  if (typeof object1 === object1 && typeof object2 === object2) {
    const keys1 = Object.keys(object1);
    const keys2 = Object.keys(object2);

    if (keys1.length !== keys2.length) {
      return false;
    }

    for (let key of keys1) {
      // @ts-ignore
      if (JSON.stringify(object1[key]) !== JSON.stringify(object2[key])) {
        return false;
      }
    }
    return true;
  } else {
    return JSON.stringify(object1) === JSON.stringify(object2);
  }
}
