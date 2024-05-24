<script lang="ts">
	import { Presentation, Slide, Code, Media, Vertical, Step } from '@components';
	import fastapi from "./assets/fastapi.svg";
	import { signal } from '@motion'

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
		<Slide>
			<div class="flex flex-row items-center justify-center text-8xl font-bold">
				비동기
			</div>
		</Slide>
		<Slide>
			<div class="flex flex-col items-center justify-center w-full h-full gap-y-12">
				<p class="text-5xl font-bold">1,3,5초 걸리는 작업 실행</p>
			</div>
			<div class="flex flex-col gap-8 mt-16">
				<div class="w-fit flex flex-row gap-x-16 text-black">
				  <p class="text-5xl font-semibold text-white">동기</p>
					<div class="flex flex-row">
						<div class="bg-green-400 px-8">
							1
						</div>
						<div class="bg-yellow-400 px-24">
							3
						</div>
						<div class="bg-red-400 px-40">
							5
						</div>
					</div>
				</div>
				<div class="w-fit flex flex-row gap-x-6 text-black">
				  <p class="text-5xl font-semibold text-white">비동기</p>
				  <div class="flex flex-col gap-y-2">
						<Step class="bg-green-400 px-8 w-fit">
							1
						</Step>
						<Step class="bg-yellow-400 px-24 w-fit">
							3
						</Step>
						<Step class="bg-red-400 px-40 w-fit">
							5
						</Step>
					</div>
				</div>
			</div>
		</Slide>
	</Vertical>
	<Vertical>
		<Slide>
			<Code lang="python" lines="1">
	async def root():
	  return "asdf"
			</Code>
		</Slide>
		<Slide>
			<Code lang="python" lines="1|2-5|6|8-9|10|11-12">
import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse

app = FastAPI()
url = "https://fnoa.misile.xyz/file/test/ny64issus.png"

@app.get("/external-api")
async def call_external_api():
  async with httpx.AsyncClient() as client:
    response = await client.get(url)
    return FileResponse(response)
			</Code>
		</Slide>
		<Slide>
			<Code lang="python" lines="1-2|4-5|7-10|11-15|16">
from fastapi import FastAPI
import databases

DATABASE_URL = "sqlite:///./test.db"
database = databases.Database(DATABASE_URL)

app = FastAPI()

@app.get("/users/&lbrace;user_id&rbrace;")
async def read_user(user_id: int):
  query = "SELECT * FROM users WHERE id = :user_id"
  user = await database.fetch_one(
    query=query,
    values=&lbrace;"user_id":user_id&rbrace;
  )
  return user
			</Code>
		</Slide>
	</Vertical>
</Presentation>
