import { Index, type Component } from 'solid-js';
import { ColorModeProvider, ColorModeScript} from '@kobalte/core';
import NavBar from './components/navbar';
import { Card } from './components/ui/card';
import dayjs, { unix } from "dayjs";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './components/ui/table';

const testdata = [{
  number: "00000",
  name: "a",
  pnumber: "01011112222",
  me: "I use nixos btw",
  why: ":sunglasses:",
  time: dayjs()
}, {
  number: "00000",
  name: "a",
  pnumber: "01011112222",
  me: "I use nixos btw\n".repeat(10),
  why: ":sunglasses:\n".repeat(10),
  time: dayjs()
}, {
  number: "00000",
  name: "a",
  pnumber: "01011112222",
  me: "I use nixos btw\n".repeat(10),
  why: ":sunglasses:\n".repeat(10),
  time: dayjs()
}, {
  number: "00000",
  name: "a",
  pnumber: "01011112222",
  me: "I use nixos btw\n".repeat(10),
  why: ":sunglasses:\n".repeat(10),
  time: dayjs()
}, {
  number: "00000",
  name: "a",
  pnumber: "01011112222",
  me: "I use nixos btw\n".repeat(10),
  why: ":sunglasses:\n".repeat(10),
  time: dayjs()
}, {
  number: "00000",
  name: "a",
  pnumber: "01011112222",
  me: "I use nixos btw\n".repeat(10),
  why: ":sunglasses:\n".repeat(10),
  time: dayjs()
}, {
  number: "00000",
  name: "a",
  pnumber: "01011112222",
  me: "I use nixos btw\n".repeat(10),
  why: ":sunglasses:\n".repeat(10),
  time: dayjs()
}, {
  number: "00000",
  name: "a",
  pnumber: "01011112222",
  me: "I use nixos btw\n".repeat(10),
  why: ":sunglasses:\n".repeat(10),
  time: dayjs()
}, {
  number: "00000",
  name: "a",
  pnumber: "01011112222",
  me: "I use nixos btw\n".repeat(10),
  why: ":sunglasses:\n".repeat(10),
  time: dayjs()
}, {
  number: "00000",
  name: "a",
  pnumber: "01011112222",
  me: "I use nixos btw\n".repeat(10),
  why: ":sunglasses:\n".repeat(10),
  time: dayjs()
}, {
  number: "00000",
  name: "a",
  pnumber: "01011112222",
  me: "I use nixos btw\n".repeat(10),
  why: ":sunglasses:\n".repeat(10),
  time: dayjs()
}, {
  number: "00000",
  name: "a",
  pnumber: "01011112222",
  me: "I use nixos btw\n".repeat(10),
  why: ":sunglasses:\n".repeat(10),
  time: dayjs()
}, {
  number: "00000",
  name: "a",
  pnumber: "01011112222",
  me: "I use nixos btw\n".repeat(10),
  why: ":sunglasses:\n".repeat(10),
  time: dayjs()
}, {
  number: "00000",
  name: "a",
  pnumber: "01011112222",
  me: "I use nixos btw\n".repeat(10),
  why: ":sunglasses:\n".repeat(10),
  time: dayjs()
}, {
  number: "00000",
  name: "a",
  pnumber: "01011112222",
  me: "I use nixos btw\n".repeat(10),
  why: ":sunglasses:\n".repeat(10),
  time: dayjs()
}, {
  number: "00000",
  name: "a",
  pnumber: "01011112222",
  me: "I use nixos btw\n".repeat(10),
  why: ":sunglasses:\n".repeat(10),
  time: dayjs()
}, {
  number: "00000",
  name: "a",
  pnumber: "01011112222",
  me: "I use nixos btw\n".repeat(10),
  why: ":sunglasses:\n".repeat(10),
  time: dayjs()
}]

const Admin: Component = () => {
  return (
    <div>
      <ColorModeScript />
      <ColorModeProvider>
        <div class="h-screen flex flex-col">
          <NavBar />
          <div class="flex flex-grow justify-center items-center">
            <Card class="flex h-5/6 w-fit flex-col">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>이름</TableHead>
                    <TableHead>학번</TableHead>
                    <TableHead>시간</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  <Index each={testdata}>
                    {(i) => {
                      return (<TableRow onClick={()=>console.log(i())}>
                        <TableCell>{i().name}</TableCell>
                        <TableCell>{i().number}</TableCell>
                        <TableCell>{i().time.toString()}</TableCell>
                      </TableRow>)
                    }}
                  </Index>
                </TableBody>
              </Table>
            </Card>
          </div>
        </div>
      </ColorModeProvider>
    </div>
  );
};

export default Admin;
