import { Accessor, JSX, Setter, createEffect, createSignal } from "solid-js";
import { SimpleEvent, Event } from "./interfaces";
import { AlertDialog, TextField } from "@kobalte/core";
import { convertDateToString } from "./utils";

export interface tmpfordialog {
  date: string,
  time: string
}

function dialogContent(
  title: Accessor<string>,
  setTitle: Setter<string>,
  start: Accessor<tmpfordialog>,
  setStart: Setter<tmpfordialog>,
  end: Accessor<tmpfordialog>,
  setEnd: Setter<tmpfordialog>,
  content: Accessor<string>,
  setContent: Setter<string>
) {
  return (<TextField.Root>
    <div class="w-full h-full flex flex-col ml-1 mr-1 mb-1 mt-1">
      <TextField.Input
        class="text-2xl font-bold outline-none bg-transparent dark:text-ctp-text"
        style="border-radius: 10px;"
        type="text"
        size={10}
        spellcheck={false}
        value={title()}
        onChange={(e)=>setTitle(e.target.value)}
      />
      <TextField.Label class="w-fit">시작 날짜</TextField.Label>
      <div class="text-xl text-gray-500 flex flex-row dark:text-ctp-subtext0 w-fit">
        <TextField.Input
          class="outline-none bg-transparent"
          type="date"
          value={start()["date"]}
          onchange={(e)=>setStart({
            "date": e.target.value,
            "time": start()["time"]
          })}
        />
        <TextField.Input
          class="outline-none bg-transparent"
          type="time"
          onchange={(e)=>setStart({
            "date": start()["date"],
            "time": e.target.value
          })}
        />
      </div>
      <TextField.Label class="w-fit">끝나는 날짜</TextField.Label>
      <div class="text-xl text-gray-500 flex flex-row dark:text-ctp-subtext0 w-fit">
        <TextField.Input
          class="outline-none bg-transparent"
          type="date"
          value={end()["date"]}
          onchange={(e)=>setStart({
            "date": e.target.value,
            "time": end()["time"]
          })}
        />
        <TextField.Input
          class="outline-none bg-transparent"
          type="time"
          value={end()["time"]}
          onchange={(e)=>setStart({
            "date": end()["date"],
            "time": e.target.value
          })}
        />
      </div>
      <TextField.Label class="w-fit">설명</TextField.Label>
      <TextField.TextArea
        class="bg-transparent outline-none resize-none max-h-24 dark:text-ctp-text scroll-smooth w-fit"
        spellcheck={false}
        autoResize
        value={content()}
        onchange={(e)=>setContent(e.target.value)}
      />
    </div>
  </TextField.Root>);
}

export function AlertDialogForEvent(item: SimpleEvent, comp: JSX.Element, events: Event[]) {
  const o = convertDateToString(item.org.start, item.org.end);
  let first = false;
  const [open, setOpen] = createSignal(false);
  const [title, setTitle] = createSignal(item.title);
  const [start, setStart] = createSignal({"date": o.start.date, "time": o.start.time});
  const [end, setEnd] = createSignal({"date": o.end.date, "time": o.end.time});
  const [content, setContent] = createSignal(item.content);
  createEffect(async () => {
    if (!open() && !first) {
      events.push({
        "start": start(),
        "title": title(),
        "content": content(),
        "end": end(),
        "color": "#0xc0ffee"
      })
    }
  })
  return (
    <AlertDialog.Root open={open()} onOpenChange={(b)=>{setOpen(b);first=true;}}>
      <AlertDialog.Trigger>{comp}</AlertDialog.Trigger>
      <AlertDialog.Portal>
        <AlertDialog.Overlay />
        <div class="flex fixed inset-0 z-50 items-center justify-center overlay w-full h-full bg-black bg-opacity-30">
          <AlertDialog.Content class="content glass bg-white dark:bg-ctp-overlay0">
            {dialogContent(title, setTitle, start, setStart, end, setEnd, content, setContent)}
          </AlertDialog.Content>
        </div>
      </AlertDialog.Portal>
    </AlertDialog.Root>
  );
}

export function CreateEventDialog(
  modal: Accessor<boolean>,
  setModalVisible: Setter<boolean>
) {
  const [title, setTitle] = createSignal("");
  const [start, setStart] = createSignal({"date": "", "time": ""});
  const [end, setEnd] = createSignal({"date": "", "time": ""});
  const [content, setContent] = createSignal("");
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
        onclick={(e) => {
          e.stopImmediatePropagation();
        }}
      >
        {dialogContent(title, setTitle, start, setStart, end, setEnd, content, setContent)}
      </div>
    </div>
  );
}
