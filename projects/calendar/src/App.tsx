import { For, JSX, createSignal } from "solid-js";
import { VsArrowLeft, VsArrowRight } from "solid-icons/vs";

interface EventDate {
  year: number;
  month: number;
  day: number;
  hour: number;
  minute: number;
  second: number;
}

interface Event {
  start: EventDate;
  end: EventDate;
  title: string;
  content: string;
  color: string;
}

interface SimpleEvent {
  year: number;
  month: number;
  day: number;
  title: string;
  content: string;
  color: string;
}

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
      second: 0,
    },
    end: {
      year: 2024,
      month: 1,
      day: 1,
      hour: 10,
      minute: 0,
      second: 0,
    },
    color: "c0ffee"
  },
];

function getColor(cont: string) {
  if (cont == "일") {
    return "text-red-500";
  } else if (cont == "토") {
    return "text-blue-500";
  } else {
    return "text-black";
  }
}

function dayDisplay(cont: string) {
  return (
    <div class="font-normal h-full flex" style="width: calc(100vw / 7);">
      <div
        class={`mt-auto mb-auto text-center w-full font-semibold ${getColor(
          cont
        )}`}
      >
        {cont}
      </div>
    </div>
  );
}

function getDateList(date: Date) {
  const lastDate = new Date(date.getFullYear(), date.getMonth(), 0).getDate();
  const dateList = [];
  for (let i = 1; i <= lastDate; i++) {
    dateList.push(i);
  }
  return dateList;
}

function daySingle(
  num: number | string = "",
  today: boolean = false,
  events: SimpleEvent[] | undefined = []
) {
  if (events === undefined) {
    events = [];
  }
  return (
    <div class={today ? "bg-gray-400" : ""} style="width: calc(100vw / 7)">
      <div class="mb-auto text-right mr-4 mt-2 font-semibold">{num}</div>
      <For each={events}>
        {(item) => {
          return <div class="text-black text-xl text-center" style={`background-color: #${item.color}`} oncontextmenu={(e)=>{
            e.preventDefault();
            console.log(item.title);
          }}>{item.title}</div>;
        }}
      </For>
    </div>
  );
}

function convertEventToHighlight() {
  const highlights: Record<string, SimpleEvent[]> = {};
  events.forEach((e)=>{
    let start = new Date(e.start.year, e.start.month-1, e.start.day);
    const end = new Date(e.end.year, e.end.month-1, e.end.day);
    while(start <= end) {
      const year = start.getFullYear();
      const month = start.getMonth()+1;
      const day = start.getDate();
      if (highlights[`${year}:${month}:${day}`] === undefined) {
        highlights[`${year}:${month}:${day}`] = [];
      }
      highlights[`${year}:${month}:${day}`].push({
        year: year,
        month: month+1,
        day: day,
        title: e.title,
        content: e.content,
        color: e.color
      });
      let newDate = start.setDate(start.getDate() + 1);
      start = new Date(newDate);
    }
  });
  return highlights;
}

function handlingButton(d: Date, setDateState: Function, amount: number) {
  const _d = d;
  _d.setMonth(d.getMonth() + amount);
  setDateState([_d]);
}

function day(date: Date) {
  const highlights: Record<string, SimpleEvent[]> = convertEventToHighlight();
  const fy = date.getFullYear();
  const fm = date.getMonth()+1;
  const fd = date.getDate();
  const tmcmp = fm-1 == new Date().getMonth() && fy == new Date().getFullYear();
  const _dateList: number[] = getDateList(date);
  const dateList: JSX.Element[][] = [[]];
  const prestart = new Date(fy, fm-1, 1).getDay();
  for (let i = 0; i < prestart; i++) {
    dateList[0].push(daySingle(""));
  }
  for (let i = 0; i < _dateList.length; i++) {
    let d = _dateList[i];
    if (new Date(fy, fm-1, d).getDay() == 0) {
      if (dateList[dateList.length-1].length !== 0) {dateList.push([]);}
      dateList[dateList.length-1].push(daySingle(d, tmcmp && fd == d, highlights[`${fy}:${fm}:${d}`]))
    } else {
      if (dateList.length == 0) {
        dateList.push([]);
      }
      dateList[dateList.length-1].push(daySingle(d, tmcmp && fd == d, highlights[`${fy}:${fm}:${d}`]))
    }
  }
  return (
    <For each={dateList}>
      {(item) => {
        return (
          <div
            class="flex flex-row h-1/5 border-gray-700 border-solid"
            style="border-top-width: 0.5px;"
          >
            <For each={item}>{(item) => item}</For>
          </div>
        );
      }}
    </For>
  );
}

function App() {
  const dates = ["일", "월", "화", "수", "목", "금", "토"];
  const today = new Date();
  const [date, setDate] = createSignal([today]);

  return (
    <div>
      <div class="bg-white w-screen h-screen flex flex-col">
        <div class="bg-gray-300 w-screen flex flex-col h-32">
          <div class="m-auto flex flex-row gap-2">
            <button
              class="h-full w-6"
              onClick={() => handlingButton(date()[0], setDate, -1)}
            >
              <VsArrowLeft class="h-full w-full" />
            </button>
            <div class="text-4xl font-bold">{`${date()[0].getFullYear()}.${
              date()[0].getMonth() + 1
            }`}</div>
            <button
              class="h-full w-6"
              onClick={() => handlingButton(date()[0], setDate, 1)}
            >
              <VsArrowRight class="h-full w-full" />
            </button>
          </div>
          <div class="flex flex-row w-full h-1/3">
            <For each={dates}>
              {(item) => {
                return dayDisplay(item);
              }}
            </For>
          </div>
        </div>
        <div class="h-full flex flex-col">{day(date()[0])}</div>
      </div>
    </div>
  );
}

export default App;
