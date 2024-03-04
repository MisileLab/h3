import { JSX } from "solid-js";
import NavBar from "./components/navbar";
import { ColorModeProvider, ColorModeScript } from "@kobalte/core";
import { Card, CardContent, CardHeader, CardTitle } from "./components/ui/card";
import { Col, Grid } from "./components/ui/grid";
import { Textarea } from "./components/ui/textarea";
import { Input } from "./components/ui/input";
import { Button } from "./components/ui/button";

export default function AnA(): JSX.Element {
  return (
    <div>
      <ColorModeScript />
      <ColorModeProvider>
        <div class="h-screen flex flex-col">
          <NavBar />
          <div class="flex flex-grow justify-center items-center">
            <Card class="w-fit h-fit pb-4 px-4">
              <div class="flex flex-col items-center gap-4">
                <h1 class="font-bold text-4xl mt-4">AnA 지원 폼</h1>
                <Grid cols={1}>
                  <Col span={3} class="flex flex-row gap-2">
                    {CardwithInput("학번")}
                    {CardwithInput("이름")}
                    {CardwithInput("전화번호")}
                  </Col>
                </Grid>
                <Grid cols={1} class="w-full">
                  <Col span={2} class="flex flex-col gap-4">
                    {CardwithTextArea("자기 소개(250자 제한)")}
                    {CardwithTextArea("동아리 지원 이유(250자 제한)")}
                  </Col>
                </Grid>
                <Button class="text-xl font-semibold">신청</Button>
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
    <Card class={classes}>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        {content}
      </CardContent>
    </Card>
  );
}

function CardwithInput(title: string) {
  return _CardShorcut(title, <Input type="text" />)
}

function CardwithTextArea(title: string) {
  return _CardShorcut(title, <Textarea class="resize-none scroll-smooth" />, "w-full");
}
