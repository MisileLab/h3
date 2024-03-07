import { Index, type Component, Show, createSignal, createEffect } from "solid-js";
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
import { isMobileOnly } from "mobile-device-detect";
import { User, client, getCookie, url } from "./definition";
import { gql } from "graphql-request";
dayjs.extend(utc)

const Admin: Component = () => {
  const [data, setData] = createSignal<Array<User> | undefined>(undefined);
  const [error, setError] = createSignal(false);
  createEffect(async () => {
  let key = getCookie("key");
  if (key === undefined) {
    key = "";
  }
  let tmp = await client.request(gql`
      query Query($key: String!) {
        infos(key:$key){
          name
          time
          pnumber
          portfolio
          me
          why
        }
      }
    `, {key: key})
  if (tmp["infos"] === null || tmp["infos"] === undefined) {
    console.error("invalid key");
    setError(true);
  }
  setData(tmp["infos"] as unknown as User[]);
  });
  return (
    <div>
      <ColorModeScript />
      <ColorModeProvider>
        {error() && <AlertDialog defaultOpen>
          <AlertDialogContent>
            <AlertDialogTitle>Invalid key</AlertDialogTitle>
            <AlertDialogDescription class="flex flex-col gap-2">
              No flag in here :sunglasses:
              <iframe src="https://www.youtube.com/embed/jdUXfsMTv7o?si=sgKI5w1E8tHFf35s" title="Tux" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen />
            </AlertDialogDescription>
          </AlertDialogContent>
        </AlertDialog>}
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
                      {!isMobileOnly && <TableHead>전화번호</TableHead>}
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    <Index each={data()}>
                      {(i) => {
                        return (
                          <AlertDialog>
                            <AlertDialogTrigger asChild>
                              <As component={TableRow}>
                                <TableCell>{dayjs(i().time).local().format('MM월 DD일 hh시 mm분')}</TableCell>
                                <TableCell>{i().name}</TableCell>
                                {!isMobileOnly && <TableCell>{i().pnumber}</TableCell>}
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
                                  <a href={`${url}/files/${getCookie('key')}/${i().portfolio.slice(1, i().portfolio.length-1)}`}>
                                    <h2>포트폴리오</h2>
                                  </a>
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
