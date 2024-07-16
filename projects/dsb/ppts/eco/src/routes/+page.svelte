<script lang="ts">
	import { Presentation, Slide, Code, Action } from '@animotion/core'

	let code: Code
</script>

<link href="https://cdn.jsdelivr.net/gh/sun-typeface/SUIT@2/fonts/variable/woff2/SUIT-Variable.css" rel="stylesheet">

<style>
  * {font-family: 'SUIT Variable', sans-serif;}
</style>

<Presentation options={{ history: true, transition: 'fade', controls: true, progress: true }}>
  <Slide>
    <section class="flex-col justify-center w-full h-full">
      <div class="flex w-screen h-full justify-center items-center gap-10">
        <img src="ecoProgressBefore.png" alt="" width="200%" />
        <span class="w-fit grow whitespace-nowrap">{`->`}</span>
        <img src="ecoProgressAfter.png" alt="" width="200%" />
      </div>
      <p class="text-7xl">행동을 실행하기 전에 고민을 하고,</p>
    </section>
    <section class="flex-col justify-center w-full h-full">
      <div class="flex w-screen h-full justify-center items-center gap-10">
        <img src="ecoProgressBefore.png" alt="" width="200%" />
        <span class="w-fit grow whitespace-nowrap">{`->`}</span>
        <img src="ecoProgressAfter.png" alt="" width="200%" />
      </div>
      <p class="text-7xl">그 후에 위험한 행동을 합니다.</p>
    </section>
    <section class="flex-col justify-center w-full h-full">
      <div class="flex w-screen h-full justify-center items-center gap-10">
        <img src="ecoProgressBefore.png" alt="" width="200%" />
        <span class="w-fit grow whitespace-nowrap">{`->`}</span>
        <img src="ecoProgressAfter.png" alt="" width="200%" />
      </div>
      <p class="text-7xl">사람이 위험한 행동을 하면 오염도가 늘어납니다.</p>
    </section>
    <section class="flex-col justify-center w-full h-full">
      <div class="flex w-screen h-full justify-center items-center gap-10">
        <img src="kickButton.png" alt="" width="50%" />
      </div>
      <p class="text-7xl">아이템을 사용해서 위험한 행동을 그만하게 하거나</p>
    </section>
    <section class="flex-col justify-center w-full h-full">
      <div class="flex w-screen h-full justify-center items-center gap-10">
        <img src="kickButton.png" alt="" width="50%" />
      </div>
      <p class="text-7xl">문제의 근원을 제거할 수도 있습니다.</p>
    </section>
    <section class="flex-col justify-center w-full h-full">
      <div class="flex w-screen h-full justify-center items-center gap-10">
        <img src="kickButton.png" alt="" width="50%" />
      </div>
      <p class="text-6xl">내쫓기를 사용하면 다른 사람들이 위험한 행동을 조금 더 하거나, 조금 덜 합니다.</p>
    </section>
    <section class="flex-col justify-center w-full h-full">
      <div class="flex w-screen h-full justify-center items-center gap-10">
        <img src="kickButton.png" alt="" width="50%" />
      </div>
      <p class="text-7xl">시간이 흐를 수록 사람의 수와 행동 빈도가 증가합니다.</p>
    </section>
    <section class="flex-col justify-center w-full h-screen">
      <div class="flex w-screen h-full justify-center items-center">
        <img src="ecoProgressFull.png" alt="" width="50%" />
      </div>
      <p class="text-7xl">오염도가 일정 수준에 다다르면 패배합니다.</p>
    </section>
  </Slide>
  <Slide>
    <section class="flex-col justify-center w-full h-screen">
      <div class="flex w-screen h-full justify-center items-center">
<Code
bind:this={code}
code={`  using System;
  using X;
  using Y;`} lang="csharp" theme="catppuccin-mocha"/>
        <Action do={() => code.selectLines`2`} />
        <Action do={() => code.selectLines`1`} />
        <Action do={() => code.selectLines`3`} />
      </div>
    </section>
  </Slide>
  <Slide>
    <section class="flex-col justify-center w-full h-screen">
      <div class="flex w-screen h-full justify-center items-center">
<Code
bind:this={code}
code={`  using System;
  using X;
  using Y;`} lang="csharp" theme="catppuccin-mocha"/>
        <Action do={() => code.selectLines`2`} />
        <Action do={() => code.selectLines`1`} />
        <Action do={() => code.selectLines`3`} />
      </div>
    </section>
  </Slide>
</Presentation>
