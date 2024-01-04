import { Accessor, For, JSX, Setter, Show, createSignal } from "solid-js";
import { VsArrowLeft, VsArrowRight, VsEdit, VsTrash } from "solid-icons/vs";
import { Transition } from "solid-transition-group";
import { EventDate, SimpleEvent } from "./interfaces";
import { getColor, convertEventToHighlight, handlingButton } from "./utils";
import { ContextMenu, AlertDialog, TextField } from "@kobalte/core";

function convertDateToString(ios: EventDate, ioe: EventDate) {
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

function AlertDialogForEvent(item: SimpleEvent, comp: JSX.Element) {
  const o = convertDateToString(item.org.start, item.org.end);
  return (
    <AlertDialog.Root>
      <AlertDialog.Trigger>{comp}</AlertDialog.Trigger>
      <AlertDialog.Portal>
        <AlertDialog.Overlay />
        <div class="flex fixed inset-0 z-50 items-center justify-center overlay w-full h-full bg-black bg-opacity-30">
          <AlertDialog.Content class="content glass bg-white dark:bg-ctp-overlay0">
            <TextField.Root>
              <div class="w-full h-full flex flex-col ml-1 mr-1 mb-1 mt-1">
                <TextField.Input
                  class="text-2xl font-bold outline-none bg-transparent dark:text-ctp-text"
                  style="border-radius: 10px;"
                  type="text"
                  size={10}
                  spellcheck={false}
                  value={item.title}
                />
                <TextField.Label class="w-fit">시작 날짜</TextField.Label>
                <div class="text-xl text-gray-500 flex flex-row dark:text-ctp-subtext0 w-fit">
                  <TextField.Input
                    class="outline-none bg-transparent"
                    type="date"
                    value={o["start"]["date"]}
                  />
                  <TextField.Input
                    class="outline-none bg-transparent"
                    type="time"
                    value={o["start"]["time"]}
                  />
                </div>
                <TextField.Label class="w-fit">끝나는 날짜</TextField.Label>
                <div class="text-xl text-gray-500 flex flex-row dark:text-ctp-subtext0 w-fit">
                  <TextField.Input
                    class="outline-none bg-transparent"
                    type="date"
                    value={o["end"]["date"]}
                  />
                  <TextField.Input
                    class="outline-none bg-transparent"
                    type="time"
                    value={o["end"]["time"]}
                  />
                </div>
                <TextField.Label class="w-fit">설명</TextField.Label>
                <TextField.TextArea
                  class="bg-transparent outline-none resize-none max-h-24 dark:text-ctp-text scroll-smooth w-fit"
                  spellcheck={false}
                  autoResize
                  value={item.content}
                />
              </div>
            </TextField.Root>
          </AlertDialog.Content>
        </div>
      </AlertDialog.Portal>
    </AlertDialog.Root>
  );
}

// i'll combine with AlertDialogForEvent and this function
function CreateEventDialog(
  modal: Accessor<boolean>,
  setModalVisible: Setter<boolean>
) {
  return (
    <div
      class="flex fixed inset-0 z-50 items-center justify-center overlay w-full h-full bg-black bg-opacity-30 tmp"
      onclick={(e) => {
        e.stopImmediatePropagation();
        setModalVisible(false);
      }}
    >
      <div
        class={`content glass bg-white dark:bg-ctp-overlay0`}
        data-expanded={modal()}
      >
        <TextField.Root>
          <div class="w-full h-full flex flex-col ml-1 mr-1 mb-1 mt-1">
            <TextField.Input
              class="text-2xl font-bold outline-none bg-transparent dark:text-ctp-text"
              style="border-radius: 10px;"
              type="text"
              size={10}
              spellcheck={false}
            />
            <TextField.Label class="w-fit">시작 날짜</TextField.Label>
            <div class="text-xl text-gray-500 flex flex-row dark:text-ctp-subtext0 w-fit">
              <TextField.Input
                class="outline-none bg-transparent"
                type="date"
              />
              <TextField.Input
                class="outline-none bg-transparent"
                type="time"
              />
            </div>
            <TextField.Label class="w-fit">끝나는 날짜</TextField.Label>
            <div class="text-xl text-gray-500 flex flex-row dark:text-ctp-subtext0 w-fit">
              <TextField.Input
                class="outline-none bg-transparent"
                type="date"
              />
              <TextField.Input
                class="outline-none bg-transparent"
                type="time"
              />
            </div>
            <TextField.Label class="w-fit">설명</TextField.Label>
            <TextField.TextArea
              class="bg-transparent outline-none resize-none max-h-24 dark:text-ctp-text scroll-smooth w-fit"
              spellcheck={false}
              autoResize
            />
          </div>
        </TextField.Root>
      </div>
    </div>
  );
}

function ContextMenuForEvent(item: SimpleEvent, comp: JSX.Element) {
  const o = convertDateToString(item.org.start, item.org.end);
  return (
    <ContextMenu.Root>
      <ContextMenu.Trigger>{comp}</ContextMenu.Trigger>
      <ContextMenu.Portal>
        <ContextMenu.Content
          class="bg-white dark:bg-ctp-overlay0 glass content outline-none"
          onclick={(e) => {
            e.stopImmediatePropagation();
          }}
        >
          <div class="flex-col flex mr-1">
            <div class="text-xl font-bold dark:text-ctp-text">{`이름: ${item.title}`}</div>
            <div class="text-lg text-gray-500 dark:text-ctp-subtext1">{`시작: ${o["start"]["full"]}`}</div>
            <div class="text-lg text-gray-500 dark:text-ctp-subtext1">{`끝: ${o["end"]["full"]}`}</div>
            <div class="flex flex-row-reverse w-full mb-1">
              <button
                onClick={() => {
                  const a = confirm("정말로 삭제하시겠습니까?");
                  if (!a) {
                    return;
                  }
                }}
              >
                <VsTrash size={24} />
              </button>
              {AlertDialogForEvent(
                item,
                <div>
                  <VsEdit size={24} />
                </div>
              )}
            </div>
          </div>
        </ContextMenu.Content>
      </ContextMenu.Portal>
    </ContextMenu.Root>
  );
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
  const [modal, setModalVisible] = createSignal(false);
  if (events === undefined) {
    events = [];
  }
  return (
    <div
      style="width: calc(100vw / 7)"
      onClick={() => {
        setModalVisible(true);
      }}
    >
      <div class="mb-auto text-right mt-2 font-semibold flex flex-row-reverse">
        <div
          class={`${
            today ? "bg-red-400 dark:text-black" : ""
          } w-6 h-6 rounded-full text-center dark:text-ctp-subtext0`}
        >
          {num}
        </div>
      </div>
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
      <Transition name="trans">
        <Show when={modal()}>{CreateEventDialog(modal, setModalVisible)}</Show>
      </Transition>
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
            class="flex flex-row h-1/5 border-gray-700 dark:border-white border-solid"
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
    <div
      oncontextmenu={(e) => {
        if (e.button == 2) {
          e.preventDefault();
        }
      }}
    >
      <div class="bg-white dark:bg-ctp-surface0 w-screen h-screen flex flex-col">
        <div class="bg-gray-300 dark:bg-ctp-surface1 w-screen flex flex-col h-32">
          <div class="m-auto flex flex-row gap-2 dark:text-ctp-text">
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
