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
            <Card class="w-fit h-fit pb-4 md:px-12 mx-0 smallphone:mx-4 md:mx-0">
              <div class="flex flex-col items-center">
                <h1 class="font-bold text-4xl mt-4">AnA 지원 폼</h1>
                <Grid cols={1}>
                  <Col span={2} class="flex flex-row">
                    {CardwithInput("학번/이름", "00000/이름")}
                    {CardwithInput("전화번호")}
                  </Col>
                </Grid>
                <Grid cols={1} class="w-full">
                  <Col span={2} class="flex flex-col">
                    {CardwithTextArea("자기 소개(250자 제한)")}
                    {CardwithTextArea("동아리 지원 이유(250자 제한)")}
                    <div class="flex flex-col sm:flex-row gap-2 items-center px-6">
                      <h3 class="text-lg font-semibold leading-none tracking-tight">포트폴리오(필수 아님)</h3>
                      <Button>업로드</Button>
                    </div>
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

function CardwithInput(title: string, placeholder: string | undefined = "") {
  return _CardShorcut(title, <Input type="text" placeholder={placeholder} class="placeholder:text-transparent smp:placeholder:text-gray-700" />)
}

function CardwithTextArea(title: string) {
  return _CardShorcut(title, <Textarea class="resize-none scroll-smooth min-h-[50px] h-[50px] md:min-h-[80px]" />, "w-full");
}
