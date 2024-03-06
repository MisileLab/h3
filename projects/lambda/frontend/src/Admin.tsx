import { Index, type Component, Show } from "solid-js";
import { A } from "@solidjs/router";
import { As, ColorModeProvider, ColorModeScript } from "@kobalte/core";
import NavBar from "./components/navbar";
import { Card, CardContent } from "./components/ui/card";
import dayjs from "dayjs";
import utc from "dayjs/plugin/utc";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "./components/ui/table";
import { AlertDialog, AlertDialogContent, AlertDialogDescription, AlertDialogTitle, AlertDialogTrigger } from "./components/ui/alert-dialog";
dayjs.extend(utc)

const testdata = [];

const Admin: Component = () => {
  for (let i = 0; i < 16; i++) {
    testdata.push({
      name: "a",
      pnumber: "01011112222",
      me: "I use nixos btw",
      why: ":sunglasses:",
      time: dayjs(),
      portfolio: "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c4/NixOS_logo.svg/1280px-NixOS_logo.svg.png"
    });
  }
  testdata[testdata.length - 1].portfolio = null;
  console.log(testdata);
  return (
    <div>
      <ColorModeScript />
      <ColorModeProvider>
        <div class="h-screen flex flex-col">
          <NavBar />
          <div class="flex justify-center items-center h-full overflow-y-hidden">
            <Card class="flex h-5/6 w-fit flex-col overflow-y-scroll">
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>제출한 시간</TableHead>
                      <TableHead>학번/이름</TableHead>
                      <TableHead>전화번호</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    <Index each={testdata}>
                      {(i) => {
                        return (
                          <AlertDialog>
                            <AlertDialogTrigger asChild>
                              <As component={TableRow}>
                                <TableCell>{i().time.local().format('MM월 DD일 hh/mm/ss')}</TableCell>
                                <TableCell>{i().name}</TableCell>
                                <TableCell>{i().pnumber}</TableCell>
                              </As>
                            </AlertDialogTrigger>
                            <AlertDialogContent>
                              <AlertDialogTitle>{i().name}</AlertDialogTitle>
                              <AlertDialogDescription class="font-normal font-mono gap-2 flex flex-col">
                                <h1 class="text-2xl font-bold">자기소개</h1>
                                <p>{i().me}</p>
                                <h1 class="text-2xl font-bold">들어온 이유</h1>
                                <p>{i().why}</p>
                                <h2 class="text-xl font-bold">전화번호</h2>
                                <p>{i().pnumber}</p>
                                <Show when={i().portfolio != null && i().portfolio != undefined}>
                                  <A href={i().portfolio}>
                                    <h2>포트폴리오</h2>
                                  </A>
                                </Show>
                              </AlertDialogDescription>
                            </AlertDialogContent>
                          </AlertDialog>
                        );
                      }}
                    </Index>
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </div>
        </div>
      </ColorModeProvider>
    </div>
  );
};

export default Admin;
