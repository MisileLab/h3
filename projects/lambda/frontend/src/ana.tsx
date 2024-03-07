import { JSX, createEffect, createSignal } from "solid-js";
import NavBar from "./components/navbar";
import { ColorModeProvider, ColorModeScript } from "@kobalte/core";
import { createFileUploader } from "@solid-primitives/upload";
import { Card, CardContent, CardHeader, CardTitle } from "./components/ui/card";
import { Col, Grid } from "./components/ui/grid";
import { Textarea } from "./components/ui/textarea";
import { Input } from "./components/ui/input";
import { Button } from "./components/ui/button";
import { client, signal, url } from "./definition";
import { gql } from "graphql-request";

export default function AnA(): JSX.Element {
  const { files, selectFiles } = createFileUploader();
  const name = createSignal("");
  const pnumber = createSignal("");
  const me = createSignal("");
  const why = createSignal("");
  const [enabled, setButtonEnabled] = createSignal(true);
  createEffect(() => {
    let enabled = true;
    [name, pnumber, me, why].forEach((i) => {
      if (i[0]() == "") { enabled = false; return false; }
    });
    setButtonEnabled(enabled);
  },);
  return (
    <div>
      <ColorModeScript />
      <ColorModeProvider>
        <div class="h-screen flex flex-col">
          <NavBar />
          <div class="flex flex-grow justify-center items-center">
            <Card class="w-fit h-fit pb-4 md:px-12 mx-0 smp:mx-4 md:mx-0">
              <div class="flex flex-col items-center">
                <h1 class="font-bold text-4xl mt-4">AnA 지원 폼</h1>
                <Grid cols={1}>
                  <Col span={2} class="flex flex-row">
                    {CardwithInput("학번/이름", "00000/이름", name)}
                    {CardwithInput("전화번호", "", pnumber)}
                  </Col>
                </Grid>
                <Grid cols={1} class="w-full">
                  <Col span={2} class="flex flex-col">
                    {CardwithTextArea("자기 소개", me)}
                    {CardwithTextArea("동아리 지원 이유", why)}
                    <div class="flex flex-col sm:flex-row gap-2 items-center px-6">
                      <h3 class="text-lg font-semibold leading-none tracking-tight">포트폴리오(필수 아님)</h3>
                      <Button onClick={() => {
                        selectFiles(async ([{ source, name, size, file }]) => {
                          console.log(source, name, size, file);
                        })
                      }}>업로드</Button>
                    </div>
                  </Col>
                </Grid>
                <Button class="text-xl font-semibold mt-4" onClick={async () => {
                  const fd = new FormData();
                  console.log(files);
                  if (files()[0] === undefined) {
                    await client.request(gql`
                    mutation Query($name: String!, $pnumber: String!, $me: String!, $why: String!, $portfolio: String) {
                          send(
                            i: {
                              name:$name,
                              pnumber:$pnumber,
                              me:$me,
                              why:$why,
                              portfolio:$portfolio
                            })}
                `, { name: name[0](), me: me[0](), why: why[0](), portfolio: null, pnumber: pnumber[0]() })
                    return;
                  }
                  fd.append('file', files()[0].file);
                  await fetch(`${url}/uploadfile`, {
                    method: "POST",
                    headers: {},
                    body: fd
                  }).then(r => {
                    r.text().then(async (v) => {
                      await client.request(gql`
                            mutation Query($name: String!, $pnumber: String!, $me: String!, $why: String!, $portfolio: String) {
                                  send(
                                    i: {
                                      name:$name,
                                      pnumber:$pnumber,
                                      me:$me,
                                      why:$why,
                                      portfolio:$portfolio
                                    })}
                        `, { name: name[0](), me: me[0](), why: why[0](), portfolio: v, pnumber: pnumber[0]() })
                    })
                  }).catch(e => console.error(e));
                }} disabled={!enabled()}>신청</Button>
              </div>
            </Card>
          </div>
        </div>
      </ColorModeProvider>
    </div>
  );
}

function _CardShorcut(title: string, content: JSX.Element, classes: string | undefined = "") {
  return (
    <Card class={`${classes} border-none`}>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        {content}
      </CardContent>
    </Card>
  );
}

function CardwithInput(title: string, placeholder: string | undefined = "", signal: signal<string>) {
  return _CardShorcut(title, <Input type="text" placeholder={placeholder} class="placeholder:text-transparent smp:placeholder:text-gray-700" onchange={(self) => signal[1](self.target.value)} value={signal[0]()} />)
}

function CardwithTextArea(title: string, signal: signal<string>) {
  return _CardShorcut(title, <Textarea class="resize-none scroll-smooth min-h-[50px] h-[50px] md:min-h-[80px]" onchange={(self) => signal[1](self.target.value)} value={signal[0]()} />, "w-full");
}
