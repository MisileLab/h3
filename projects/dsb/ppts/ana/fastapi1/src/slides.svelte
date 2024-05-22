<script lang="ts">
	import { Presentation, Slide, Code, Media, Vertical, Step } from '@components';
	import fastapi from "./assets/fastapi.svg";
	import { signal } from '@motion'

	let django = signal(0);
	let flask = signal(0);
	let fastapiSignal = signal(0);

	async function animateDjango() { await django.to(77190) }
	async function animateFlask() { await flask.to(66598) }
	async function animateFastAPI() { await fastapiSignal.to(71722) }
</script>
<link href="https://cdn.jsdelivr.net/gh/sun-typeface/SUIT/fonts/variable/woff2/SUIT-Variable.css" rel="stylesheet">

<style>
    * {font-family: 'SUIT Variable', sans-serif;}
</style>

<Presentation>
	<Slide animate>
		<div class="flex flex-row gap-2 w-full items-center justify-center">
			<Media type="img" preload src={fastapi} width={100}/>
			<h1 class="font-bold text-8xl">FastAPI</h1>
		</div>
	</Slide>
	<Vertical>
		<Slide on:out={()=>{django.reset();flask.reset();fastapiSignal.reset()}}>
			<div class="flex flex-col gap-8 tabular-nums">
				<h1 class="font-bold text-6xl">star 수</h1>
				<div class="flex flex-row gap-12 w-full items-center justify-center">
					<div class="flex flex-col gap-8">
						<h1 class="font-semibold text-6xl">Django</h1>
						<django font-size={48}>{$django.toFixed(0)}</django>
						<Step on:in={animateDjango} on:out={django.reset}/>
					</div>
					<div class="flex flex-col gap-8">
						<h1 class="font-semibold text-5xl">Flask</h1>
						<flask font-size={48}>{$flask.toFixed(0)}</flask>
						<Step on:in={animateFlask} on:out={flask.reset} />
					</div>
					<div class="flex flex-col gap-8">
						<h1 class="font-semibold text-5xl">FastAPI</h1>
						<fastapiSignal font-size={48}>{$fastapiSignal.toFixed(0)}</fastapiSignal>
						<Step on:in={animateFastAPI} on:out={fastapiSignal.reset} />
					</div>
				</div>
			</div>
		</Slide>
		<Slide animate>
			<div class="flex flex-col gap-8 tabular-nums">
				<h1 class="font-bold text-6xl">Benchmark (req/s)</h1>
				<div class="flex flex-row gap-12 w-full items-center justify-center">
					<div class="flex flex-col gap-8">
						<h1 class="font-semibold text-6xl">Django</h1>
						<django font-size={48}>{$django.toFixed(0)}</django>
						<Step on:in={async ()=>{await django.to(19402)}} on:out={django.reset}/>
					</div>
					<div class="flex flex-col gap-8">
						<h1 class="font-semibold text-5xl">Flask</h1>
						<flask font-size={48}>{$flask.toFixed(0)}</flask>
						<Step on:in={async ()=>{await flask.to(44033)}} on:out={flask.reset} />
					</div>
					<div class="flex flex-col gap-8">
						<h1 class="font-semibold text-5xl">FastAPI</h1>
						<fastapiSignal font-size={48}>{$fastapiSignal.toFixed(0)}</fastapiSignal>
						<Step on:in={async ()=>{await fastapiSignal.to(62020)}} on:out={fastapiSignal.reset} />
					</div>
				</div>
			</div>
		</Slide>
	</Vertical>
	<Vertical>
		<Slide>
			<div class="flex flex-col gap-8">
				<h1 class="font-bold text-6xl">Fastapi 특징</h1>
				<p class="text-5xl">- 기능이 적어서 빠름</p>
				<p class="text-5xl">- 메인 언어가 C라 빠름</p>
				<p class="text-5xl">- 비동기 기본</p>
			</div>
		</Slide>
		<Slide>
			<div class="flex flex-col gap-8">
				<p class="text-5xl">openapi = 대충 api 문서 표준</p>
				<p class="text-4xl">어차피 알아서 자동생성함</p>
			</div>
		</Slide>
	</Vertical>
	<Vertical>
		<Slide>
			<Code lang="python" lines="1|2|4-6|4">
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
async def root():
	return 'hi'
			</Code>
		</Slide>
		<Slide>
			<div class="flex flex-col gap-2 items-center justify-center">
				<p class="text-4xl font-bold">GET - 조회</p>
				<p class="text-4xl font-bold">POST - 변경</p>
				<p class="text-4xl font-bold">DELETE - 삭제</p>
			</div>
		</Slide>
	</Vertical>
</Presentation>
