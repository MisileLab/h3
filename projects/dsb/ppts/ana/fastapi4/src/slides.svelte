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
	    <Code lang="python">
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/")
async def read_items(q: str | None = None):
  if q is None:
    return "None"
  return q
	    </Code>
	  </Slide>
	  <Slide animate>
	    <Code lang="python" lines="1-3|8">
from fastapi import FastAPI, Query

from typing import Annotated

app = FastAPI()

@app.get("/items/")
async def read_items(q: Annotated[str | None, Query(max_length=10)] = None):
  if q is None:
    return "None"
  return q
	    </Code>
	  </Slide>
	  <Slide animate>
	    <Code lang="python" lines="8">
from fastapi import FastAPI, Query

from typing import Annotated

app = FastAPI()

@app.get("/items/")
async def read_items(q: Annotated[str | None, Query(min_length=5, max_length=10)] = None):
  if q is None:
    return "None"
  return q
      </Code>
    </Slide>
    <Slide>
      <Media src="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fstatic.ian.pw%2Fimages%2Fnixos-logo.png&f=1&nofb=1&ipt=86d5950a53c0b6fc39578c4b5c4b2aa2015aa0c567a2f9a5cc632cbc237f46ec&ipo=images" type="img"/>
      <Step>
        <p class="text-4xl">/nix/store/<span class="text-blue-500">sha256hash</span>-<span class="text-red-500">package</span>-<span class="text-green-500">version</span></p>
        <p class="text-4xl">/nix/store/<span class="text-blue-500">1iscdpbd3x9x3s3s25jd5ppl7yra0b77</span>-<span class="text-red-500">perl</span>-<span class="text-green-500">5.38.2</span></p>
      </Step>
    </Slide>
    <Slide>
      <Code lang="python" lines="8">
from fastapi import FastAPI, Query

from typing import Annotated

app = FastAPI()

@app.get("/items/")
async def read_items(q: Annotated[str | None, Query(pattern=r"\/nix\/store\/[a-z|0-9]+-.+-.+")] = None):
  if q is None:
    return "None"
  return q
      </Code>
    </Slide>
    <Slide animate>
      <Code lang="python" lines="8">
from fastapi import FastAPI, Query

from typing import Annotated

app = FastAPI()

@app.get("/items/")
async def read_items(q: Annotated[str | None, Query(max_length=10)] = None):
  if q is None:
    return "None"
  return q
      </Code>
    </Slide>
    <Slide animate>
      <Code lang="python" lines="8">
from fastapi import FastAPI, Query

from typing import Annotated

app = FastAPI()

@app.get("/items/")
async def read_items(q: Annotated[str, Query(max_length=10)]):
  if q is None:
    return "None"
  return q
      </Code>
    </Slide>
    <Slide animate>
      <Code lang="python" lines="8">
from fastapi import FastAPI, Query

from typing import Annotated

app = FastAPI()

@app.get("/items/")
async def read_items(q: Annotated[str | None, Query(max_length=10)]):
  if q is None:
    return "None"
  return q
      </Code>
    </Slide>
    <Slide animate>
      <Notes>url?q=1&q=2</Notes>
      <Code lang="python" lines="8">
from fastapi import FastAPI, Query

from typing import Annotated

app = FastAPI()

@app.get("/items/")
async def read_items(q: Annotated[list[str] | None, Query()]):
  if q is None:
    return "None"
  return q
      </Code>
    </Slide>
    <Slide animate>
      <Notes>include_in_schema True and False test</Notes>
      <Code lang="python" lines="8-11">
from fastapi import FastAPI, Query

from typing import Annotated

app = FastAPI()

@app.get("/items/")
async def read_items(q: Annotated[str, Query(
  max_length=10, title="macbook m10 pro", alias="m10",
  deprecated=True, include_in_schema=True
)]):
  if q is None:
    return "None"
  return q
      </Code>
    </Slide>
    <Slide>
      <Code lang="python">
from fastapi import FastAPI, Path, Query

from typing import Annotated

app = FastAPI()

@app.get("/items/&lbrace;item&rbrace;")
async def read_items(
  item: Annotated[int, Path(title="a")],
  q: Annotated[str | None, Query()] = None,
):
  return [item, q]
      </Code>
    </Slide>
    <Slide animate>
      <Notes>try lt=10.5</Notes>
      <Code lang="python" lines="10">
from fastapi import FastAPI, Path, Query

from typing import Annotated

app = FastAPI()

@app.get("/items/&lbrace;item&rbrace;")
async def read_items(
  item: Annotated[int, Path(title="a")],
  q: Annotated[float, Query(gt=0, lt=10)],
):
  return [item, q]
      </Code>
    </Slide>
    <Slide>
      <Notes>&lbrace;club: clubcontent, person: personcontent&rbrace;</Notes>
      <Code lang="python" lines="23">
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

class Person(BaseModel):
  club: Club
  name: str

@app.post("/verify")
async def verify(club: ClubModel, person: Person):
  if club.club is Club.ana:
    return f"You are AnA, name is &lcub;club.name&rcub;"
  return "You are not AnA"
      </Code>
    </Slide>
    <Slide animate>
      <Code lang="python" lines="23-25">
from fastapi import FastAPI, Body
from enum import Enum
from pydantic import BaseModel

from typing import Annotated

app = FastAPI()

class Club(str, Enum):
  ana = "ana"
  tapie = "tapie"
  creal = "creal"

class ClubModel(BaseModel):
  club: Club
  count: int

class Person(BaseModel):
  club: Club
  name: str

@app.post("/verify")
async def verify(club: ClubModel, person: Person, comment: Annotated[str, Body()]):
  if club.club is Club.ana:
    return f"You are AnA, name is &lcub;club.name&rcub; and &lcub;person.name&rcub;"
  return "You are not AnA"
      </Code>
    </Slide>
    <Slide animate>
      <Code lang="python" lines="19-21">
from fastapi import FastAPI, Body
from enum import Enum
from pydantic import BaseModel

from typing import Annotated

app = FastAPI()

class Club(str, Enum):
  ana = "ana"
  tapie = "tapie"
  creal = "creal"

class ClubModel(BaseModel):
  club: Club
  count: int

@app.post("/verify")
async def verify(club: Annotated[ClubModel, Body(embed=true)]):
  if club.club is Club.ana:
    return f"You are AnA, name is &lcub;club.name&rcub;"
  return "You are not AnA"
      </Code>
    </Slide>
  </Vertical>
</Presentation>
