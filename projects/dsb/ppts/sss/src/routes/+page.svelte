<script lang="ts">
  import { Presentation, Slide, Action } from '@animotion/core'

  let shamir: HTMLElement
  let secretKeySharing: HTMLElement
  let image: HTMLImageElement
  let bars: HTMLDivElement[] = [];
  let colors = ['bg-green-400', 'bg-purple-400', 'bg-red-400']
</script>

<div class="text-blue-400 text-green-400 underline border-black border-r-8' bg-green-400 bg-purple-400 bg-red-400" style="display: none;"></div>

<Presentation options={{history: true, transition: 'none', progress: false, controls: false}}>
  <Slide class="h-full place-content-center place-items-center">
    <h1 class="text-8xl">
      <span bind:this={shamir}>Shamir</span>
      's <span bind:this={secretKeySharing}>secret sharing</span>
    </h1>
    <Action do={()=>shamir.classList.add('text-green-400')} undo={()=>shamir.classList.remove('text-green-400')} />
    <Action do={()=>secretKeySharing.classList.add('text-blue-400')} undo={()=>secretKeySharing.classList.remove('text-blue-400')} />
  </Slide>
  <Slide class="h-full place-content-center place-items-center">
    <div class="flex flex-row w-screen h-screen place-content-center place-items-center">
      <div bind:this={bars[0]} class="bg-blue-400 w-1/7 h-1/4"></div>
      <div bind:this={bars[1]} class="bg-blue-400 w-1/7 h-1/4"></div>
      <div bind:this={bars[2]} class="bg-blue-400 w-1/7 h-1/4"></div>
      <div bind:this={bars[3]} class="bg-blue-400 w-1/7 h-1/4"></div>
    </div>
    <Action do={()=>{
      for (let bar of bars.slice(0, 3)) {
        bar.classList.add('border-r-8')
        bar.classList.add('border-black')
      }
    }} undo={()=>{
      for (let bar of bars.slice(0, 3)) {
        bar.classList.remove('border-r-8')
        bar.classList.remove('border-black')
      }
    }}/>
    <Action do={()=>{
      for (let i=1;i<4;i++) {
        bars[i].classList.remove('bg-blue-400')
        bars[i].classList.add(colors[i-1])
      }
    }} undo={()=>{
      for (let i=1;i<4;i++) {
        bars[i].classList.remove(colors[i-1])
        bars[i].classList.add('bg-blue-400')
      }
    }}/>
  </Slide>
  <Slide class="h-full place-content-center place-items-center">
    <div class="mb-20">
      <p class="text-5xl text-gray-500">{`\\[(a_0=S|a_1,...,a_{k-1}>=1)\\]`}</p>
      <p class="text-7xl">{`\\[f(x)=a_0+a_1x+a_2x^2+...+a_{k-1}x^{k-1}\\]`}</p>
    </div>
  </Slide>
  <Slide class="place-content-center place-items-center">
    <div class="h-screen w-full py-20">
      <img bind:this={image} src="/shamirSimple.png" class="h-full" alt="shamir" />
    </div>
    <Action do={()=>image.src="/shamirComplex.png"} undo={()=>image.src="/shamirSimple.png"}/>
  </Slide>
  <Slide class="h-full place-content-center place-items-center">
    <div class="mb-20">
      <p class="text-7xl">{`\\[f(x)=a_0+a_1x+a_2x^2+...+a_{k-1}x^{k-1}\\]`}</p>
      <p class="text-7xl">{`\\[D_x=(x+1, f(x+1))\\]`}</p>
    </div>
  </Slide>
  <Slide class="h-full place-content-center place-items-center">
    <h1 class="text-8xl">만약에 <span class="text-green-400">100</span>명에게 점을 준다면?</h1>
  </Slide>
  <Slide class="h-full place-content-center place-items-center">
    <p class="text-8xl">{`\\[(x, f(x)\\space mod \\space p)\\]`}</p>
    <p class="text-6xl text-gray-500">{`\\[(p=prime,p>S,p>n)\\]`}</p>
  </Slide>
  <Slide class="place-content-center place-items-center">
    <div class="h-screen w-full py-20">
      <img src="/modular.png" class="h-full" alt="shamir" />
    </div>
  </Slide>
  <Slide class="place-content-start place-items-start my-12 mx-12">
    <h1 class="text-8xl">자료</h1>
    <ul class="list-disc place-items-start">
      <li><a href="https://ehdvudee.tistory.com/27">ehdvudee(Tistory)</a></li>
      <li><a href="https://en.wikipedia.org/wiki/Shamir%27s_secret_sharing">Shamir's secret sharing(Wikipedia)</a></li>
      <li><a href="https://medium.com/@keylesstech/a-beginners-guide-to-shamir-s-secret-sharing-e864efbf3648">Keyless Technologies(Medium)</a></li>
      <li><a href="https://m.blog.naver.com/luexr/223238472229">Luexr(Blog)</a></li>
    </ul>
  </Slide>
</Presentation>
