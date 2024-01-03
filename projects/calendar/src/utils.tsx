import { Event, SimpleEvent } from './interfaces';

const events: Event[] = [
  {
    title: "asd",
    content: "asdf",
    start: {
      year: 2023,
      month: 12,
      day: 30,
      hour: 10,
      minute: 0,
    },
    end: {
      year: 2024,
      month: 1,
      day: 1,
      hour: 10,
      minute: 0,
    },
    color: "c0ffee",
  },
];

export function convertEventToHighlight() {
  const highlights: Record<string, SimpleEvent[]> = {};
  events.forEach((e) => {
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
