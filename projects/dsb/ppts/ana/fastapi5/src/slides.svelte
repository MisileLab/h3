<script lang="ts">
	import { Presentation, Slide, Code, Media, Vertical } from '@components';
	import fastapi from "./assets/fastapi.svg";
  import request from "sync-request"

  const lines: string[] = ["15-16", "18-20|22-25|25", "21", "18-27", "19-21|25-30", "25-33"]
  const limit = 6
  const content: string[] = []
  for (let i = 0;i<limit;i++) {
    content.push(request("GET", `/fastapi5/${i+1}`).getBody('utf8'))
  }
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
    {#each Array(limit).keys() as k}
      <Slide animate>
        <Code lang="python" lines={lines[k]}>
          {content[k]}
        </Code>
      </Slide>
    {/each}
  </Vertical>
</Presentation>
