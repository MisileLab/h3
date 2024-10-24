<script lang="ts">
	import { Presentation, Slide, Action } from '@animotion/core'

	let shamir: HTMLElement
  let secretKeySharing: HTMLElement
  let image: HTMLImageElement
</script>

<Presentation options={{history: true}}>
	<Slide class="h-full place-content-center place-items-center">
    <h1 class="text-8xl">
      <span bind:this={shamir}>Shamir</span>
      's <span bind:this={secretKeySharing}>secret key sharing</span>
    </h1>
    <Action do={()=>shamir.classList.add('text-green-400')} undo={()=>shamir.classList.remove('text-green-400')} />
    <Action do={()=>secretKeySharing.classList.add('text-blue-400')} undo={()=>secretKeySharing.classList.remove('text-blue-400')} />
  </Slide>
  <Slide class="place-content-center place-items-center">
    <div class="h-screen w-full py-20">
      <img bind:this={image} src="/shamirSimple.png" class="h-full" alt="shamir" />
    </div>
    <Action do={()=>image.src="/shamirComplex.png"} undo={()=>image.src="/shamirSimple.png"}/>
  </Slide>
  <Slide class="h-full place-content-center place-items-center">
    <p class="text-7xl">{`\\[f(x)=a_{0}+a_{1}x+a_{2}x_{2}+...+a_{k-1}x_{k-1}\\]`}
    <p class="text-7xl">{`\\[(a_{0}=S|a_{1},...,a_{k-1}>=1)\\]`}
  </Slide>
  <Slide class="h-full">
    <p class="text-6xl">{`\\[(a_{0}=1234,a_{1}=166,a_{2}=94,n=6,k=3)\\]`}</p>
  </Slide>
  <Slide class="h-full place-content-center place-items-center">
    <h1 class="text-8xl">만약에 <span class="text-green-400">100</span>명에게 점을 준다면?</h1>
  </Slide>
</Presentation>
