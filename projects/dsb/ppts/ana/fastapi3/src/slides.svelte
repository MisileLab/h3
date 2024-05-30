<script lang="ts">
	import { Presentation, Slide, Code, Media, Vertical, Step } from '@components';
	import fastapi from "./assets/fastapi.svg";
	import Notes from '@lib/components/notes.svelte';
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
      <Notes>
        fastapi 실행 실습
      </Notes>
      <p class="text-7xl font-bold mb-6">fastapi</p>
      <Code lang="python">
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
async def root():
  return &lcub;"message": "Hello, World!"&rcub;
      </Code>
      <Code lang="" class="mt-6">fastapi dev main.py</Code>
      <Code lang="" class="mt-6">fastapi run main.py</Code>
    </Slide>
    <Slide animate>
      <div class="flex flex-row gap-x-12">
        <div class="flex flex-col gap-y-6">
          <p class="text-6xl">/docs</p>
          <Step>
            <Media src="https://fastapi.tiangolo.com/img/index/index-01-swagger-ui-simple.png" type="img"/>
          </Step>
        </div>
        <div class="flex flex-col gap-y-6">
          <p class="text-6xl">/redoc</p>
          <Step>
            <Media src="https://fastapi.tiangolo.com/img/index/index-02-redoc-simple.png" type="img"/>
          </Step>
        </div>
      </div>
    </Slide>
    <Slide animate>
      <Code lang="python" lines="1|2|4|5|6">
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
async def root():
  return &lcub;"message": "Hello, World!"&rcub;
      </Code>
    </Slide>
    <Slide animate>
      <Notes>
        item_id에 1, a(잘못된 값) 넣는 거 실습
        docs 들어가보기
      </Notes>
      <Code lang="python" lines="4|5|6">
from fastapi import FastAPI
app = FastAPI()

@app.get("/items/&lcub;item&rcub;")
async def read_item(item_id: int):
  return &lcub;"id": item_id&rcub;
      </Code>
    </Slide>
    <Slide animate>
      <Notes>
        겹치면 첫번째 함수만 작동됨
      </Notes>
      <Code lang="python" lines="4-6|8-10">
from fastapi import FastAPI
app = FastAPI()

@app.get("/clubs")
async def clubs():
  return ["AnA", "tapie", "creal"]

@app.get("/clubs")
async def clubs2():
  return ["AnA", "rg", "zeropen", "iwop", "applepi", "edcan"]
      </Code>
    </Slide>
    <Slide animate>
      <Notes>
        Enum 안에 없으면 자동으로 에러 남
        직접 path쳐서 에러 내보기
        docs 들어가서 실습해보기
      </Notes>
      <Code lang="python" lines="5-8|10-11">
from fastapi import FastAPI
from enum import Enum
app = FastAPI()

class Club(str, Enum):
  ana = "ana"
  tapie = "tapie"
  creal = "creal"

@app.get("/verify/&lcub;club&rcub;")
async def verify(club: Club):
  if club is Club.ana:
    return "You are AnA"
  return "You are not AnA"
      </Code>
    </Slide>
    <Slide animate>
      <Code lang="python" lines="10-11">
from fastapi import FastAPI
from enum import Enum
app = FastAPI()

class Club(str, Enum):
  ana = "ana"
  tapie = "tapie"
  creal = "creal"

@app.get("/verify/&lcub;club:club_type&rcub;")
async def verify(club: Club):
  if club is Club.ana:
    return "You are AnA"
  return "You are not AnA"
      </Code>
    </Slide>
    <Slide animate>
      <Notes>
        http://test?n=on,1,True,true,yes 모두 작동하는거 시험
      </Notes>
      <Code lang="python" lines="10-12">
from fastapi import FastAPI
app = FastAPI()

class Club(str, Enum):
  ana = "ana"
  tapie = "tapie"
  creal = "creal"

@app.get("/verify/&lcub;club&rcub;")
async def verify(club: Club = Club.ana, n: bool = True):
  if club is Club.ana:
    return f"You are AnA, n is &lcub;n&rcub;"
  return "You are not AnA"
      </Code>
    </Slide>
    <Slide animate>
      <Notes>
        http://test?n=on,1,True,true,yes 모두 작동하는거 시험
        없으면 에러 나는거 시험
      </Notes>
      <Code lang="python" lines="10">
from fastapi import FastAPI
app = FastAPI()

class Club(str, Enum):
  ana = "ana"
  tapie = "tapie"
  creal = "creal"

@app.get("/verify/&lcub;club&rcub;/n/&lcub;n&rcub;")
async def verify(club: Club = Club.ana, n: int):
  if club is Club.ana:
    return f"You are AnA, n is &lcub;n&rcub;"
  return "You are not AnA"
      </Code>
    </Slide>
    <Slide animate>
      <Notes>
        http://test?n=on,1,True,true,yes 모두 작동하는거 시험
        없으면 에러 나는거 시험
      </Notes>
      <Code lang="python" lines="10">
from fastapi import FastAPI
app = FastAPI()

class Club(str, Enum):
  ana = "ana"
  tapie = "tapie"
  creal = "creal"

@app.get("/verify/&lcub;club&rcub;/n/&lcub;n&rcub;")
async def verify(club: Club = Club.ana, n: int = 0):
  if club is Club.ana:
    return f"You are AnA, n is &lcub;n&rcub;"
  return "You are not AnA"
      </Code>
    </Slide>
    <Slide animate>
      <Notes>
        request body http로 오는 거 설명
        docs 들어가서 실습
      </Notes>
      <Code lang="python" lines="3|6-9|11-13|15-19">
from fastapi import FastAPI
from enum import Enum
from pydantic import BaseModel
app = FastAPI()

class Club(str, Enum):
  ana = "ana"
  tapie = "tapie"
  creal = "creal"

class ClubModel(BaseModel):
  club: Club
  count: int

@app.get("/verify")
async def verify(club: ClubModel):
  if club.club is Club.ana:
    return f"You are AnA, count is &lcub;club.count&rcub;"
  return "You are not AnA"
      </Code>
    </Slide>
  </Vertical>
</Presentation>
