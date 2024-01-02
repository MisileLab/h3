import { For, JSX, createSignal } from "solid-js";
import { VsArrowLeft, VsArrowRight } from "solid-icons/vs";
import { SimpleEvent } from "./interfaces";
import { getColor, convertEventToHighlight, handlingButton } from "./utils";
import { ContextMenu, Button } from "@kobalte/core";

function ContextMenuForEvent(item: SimpleEvent, comp: JSX.Element) {
  const ios = item.org.start;
  const ioe = item.org.end;
  // 이벤트 변경, 삭제, 확인
  return (
    <ContextMenu.Root>
      <ContextMenu.Trigger>
        {comp}
      </ContextMenu.Trigger>
      <ContextMenu.Portal>
        <ContextMenu.Content class="bg-white rounded-lg border-gray-400 border-solid border-2">
          <div class="flex-col flex">
            <div class="text-xl font-bold">{`이름: ${item.title}`}</div>
            <div class="text-lg text-gray-500">{`시작: ${ios.year}.${ios.month}.${ios.day} ${ios.hour}:${ios.minute}:${ios.second}`}</div>
            <div class="text-lg text-gray-500">{`끝: ${ioe.year}.${ioe.month}.${ioe.day} ${ioe.hour}:${ioe.minute}:${ioe.second}`}</div>
            <div class="text-lg">{`설명: ${item.content}`}</div>
            <div class="flex flex-row-reverse w-full">
              <Button.Root class="border-solid border-red-500 border-2 rounded-lg" onClick={()=>{
                const a = confirm("정말로 삭제하시겠습니까?");
                if (!a) { return; }
              }}>삭제</Button.Root>
              <Button.Root class="border-solid border-gray-500 border-2 rounded-lg mr-2">변경</Button.Root>
            </div>
          </div>
        </ContextMenu.Content>
      </ContextMenu.Portal>
    </ContextMenu.Root>
  )
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
          const a = (
            <div
              class="text-black text-xl text-center"
              style={`background-color: #${item.color}`}
            >
              {item.title}
            </div>
          );
          return ContextMenuForEvent(item, a);
        }}
      </For>
    </div>
  );
}

function day(date: Date) {
  const highlights: Record<string, SimpleEvent[]> = convertEventToHighlight();
  const fy = date.getFullYear();
  const fm = date.getMonth() + 1;
  const fd = date.getDate();
  const tmcmp =
    fm - 1 == new Date().getMonth() && fy == new Date().getFullYear();
  const _dateList: number[] = getDateList(date);
  const dateList: JSX.Element[][] = [[]];
  const prestart = new Date(fy, fm - 1, 1).getDay();
  for (let i = 0; i < prestart; i++) {
    dateList[0].push(daySingle(""));
  }
  for (let i = 0; i < _dateList.length; i++) {
    let d = _dateList[i];
    if (new Date(fy, fm - 1, d).getDay() == 0) {
      if (dateList[dateList.length - 1].length !== 0) {
        dateList.push([]);
      }
      dateList[dateList.length - 1].push(
        daySingle(d, tmcmp && fd == d, highlights[`${fy}:${fm}:${d}`])
      );
    } else {
      if (dateList.length == 0) {
        dateList.push([]);
      }
      dateList[dateList.length - 1].push(
        daySingle(d, tmcmp && fd == d, highlights[`${fy}:${fm}:${d}`])
      );
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
