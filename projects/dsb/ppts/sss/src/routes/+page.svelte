<script lang="ts">
	import { Presentation, Slide, Code, Action } from '@animotion/core'

	let shamir: HTMLElement
  let secretKeySharing: HTMLElement
	let code: ReturnType<typeof Code>
</script>

<!--
https://www.geeksforgeeks.org/shamirs-secret-sharing-algorithm-cryptography/
https://medium.com/@keylesstech/a-beginners-guide-to-shamir-s-secret-sharing-e864efbf3648
https://ehdvudee.tistory.com/27
-->

<Presentation options={{history: true}}>
  <div class="text-green-400 text-blue-400 underline" style="display: none;" />
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
      <img src="/shamirSimple.png" class="h-full" />
    </div>
  </Slide>
</Presentation>
