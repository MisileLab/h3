import { Accessor, JSX, Setter } from "solid-js";
import { SimpleEvent } from "./interfaces";
import { AlertDialog, TextField } from "@kobalte/core";
import { convertDateToString } from "./utils";

export function AlertDialogForEvent(item: SimpleEvent, comp: JSX.Element) {
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

export function CreateEventDialog(
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
        onclick={(e)=>{e.stopImmediatePropagation();}}
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

