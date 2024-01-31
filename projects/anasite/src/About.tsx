import { ColorModeProvider, ColorModeScript } from "@kobalte/core";
import NavBar from "./components/navbar";
import { Col, Grid } from "./components/ui/grid";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "./components/ui/card";
import { LineChart } from "~/components/ui/charts"

export default function About() {
  const chartData = {
    labels: ["2021", "2022", "2023"],
    datasets: [
      {
        label: "Frontend",
        data: [49725, 60000, 59970],
        fill: true
      }, {
        label: "Backend",
        data: [56723, 68355, 76034],
        fill: true
      }, {
        label: "FullStack",
        data: [56038, 66372, 71140],
        fill: true
      }
    ]
  };
  return (
    <div>
      <ColorModeScript/>
      <ColorModeProvider>
        <div class="h-screen flex flex-col">
          <NavBar />
          <div class="flex flex-col ml-32 mt-14 flex-grow">
            <h1 class="font-bold text-5xl">풀스택, 프로젝트, 대회</h1>
            <Grid cols={1} colsMd={3} class="w-full gap-2 mt-8 flex-grow">
              <Col span={1} spanMd={1} class="mb-20">
                <Card class="h-full">
                  <CardHeader>
                    <CardTitle class="text-3xl">선린 유일의 풀스택 동아리</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <h1 class="text-2xl font-normal">연봉으로 나오는 백엔드의 중요성</h1>
                    <LineChart data={chartData} />
                    <CardFooter class="text-gray-400">StackOverflow Survey, 단위: $</CardFooter>
                    <h1 class="text-2xl font-normal">커리큘럼</h1>
                    <p class="mt-2">시작은 서버의 기본인 리눅스부터,</p>
                    <p>데이터를 저장하는데 필수인 데이터베이스,</p>
                    <p>백엔드에 사용되는 Python과 Java,</p>
                    <p>프론트에 거의 많이 사용되는 JavaScript,</p>
                    <p>요즘 뜨고 있는 Rust까지</p>
                    <p class="text-xl font-semibold">이 모든 걸 AnA에서 배울 수 있습니다.</p>
                  </CardContent>
                </Card>
              </Col>
              <Col span={1} spanMd={1} class="mb-20">
                <Card class="h-full">
                  <CardHeader>
                    <CardTitle class="text-3xl">아이디어, 프로젝트, 모두 지원</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <h1 class="text-2xl font-normal">아이디어로 완성되는 프로젝트</h1>
                    <p class="mt-2">좋은 아이디어를 가지고 있다면 AnA에 들어와서 프로젝트를 만들어보세요.</p>
                    <p>동아리 지원비로 지원해드리며, 서류도 대신 작성해드립니다.</p>
                    <p class="text-gray-400">만약 학과에서 거절하면, 사비로 지원해드릴 수도 있습니다.</p>
                    <h1 class="text-2xl font-normal mt-2">후배만 배우는 곳이 아닌 동아리</h1>
                    <p class="mt-2">AnA에서 모르는 것이 있다면 물어봐도 됩니다.</p>
                    <p>AnA는 디스코드와 카카오톡이 존재하며, 꼭 필요한 채팅뿐만 아니라 하고 싶은 말을 하셔도 괜찮습니다.</p>
                    <h1 class="text-2xl font-normal mt-2"></h1>
                  </CardContent>
                </Card>
              </Col>
              <Col span={1} spanMd={1} class="mb-20">
                <Card class="h-full">
                  <CardHeader>
                    <CardTitle class="text-3xl">대회 준비까지</CardTitle>
                  </CardHeader>
                  <CardContent>KPI 1</CardContent>
                </Card>
              </Col>
            </Grid>
          </div>
        </div>
      </ColorModeProvider>
    </div>
  );
}
