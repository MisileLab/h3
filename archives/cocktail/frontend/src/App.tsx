import { For, createEffect, createSignal, type Component } from 'solid-js';

const App: Component = () => {
  const [msglist, setmsglist] = createSignal([]);
  const [data, setData] = createSignal("");
  const onion_url = "mjwodo5beyddc5rdz72q24ewdtgmwe2bmlll4yqhqb56nsnggyzksayd.onion";
  const name = "test";
  const pw = prompt("password");
  createEffect(async ()=>{
    await fetch(`http://${onion_url}`).then(async (a)=>{console.log(await a.text());}).catch((a)=>{console.error(`error: ${a}`)})
  });
  const w = new WebSocket(`ws://${onion_url}/chat/${pw}`);
  w.addEventListener('open', function(){
    console.log("open");
    w.send(JSON.stringify({"type": "login", "name": name}))
  })
  w.addEventListener('message', function(e){
    console.log(`message: ${e.data}`)
    setmsglist([...msglist(), JSON.parse(e.data)]);
  })
  w.addEventListener('close', function(){
    console.log("close");
  })
  return (
    <div class="flex flex-col min-w-screen min-h-screen bg-ctp-crust items-center gap-2">
      <p class="text-ctp-blue text-3xl font-bold mt-8">Cocktail</p>
      <div class="flex flex-col w-full h-fit items-center">
        <For each={msglist()}>
          {(i,_)=><p class="text-ctp-text">{`${i["msg"]} by ${i["name"]}`}</p>}
        </For>
      </div>
      <input type="text" onchange={(e)=>setData(e.target.value)} value={data()} class="text-ctp-text bg-ctp-overlay0"/>
      <button class="text-ctp-green bg-ctp-overlay0" onclick={()=>{
        console.log(data);
        w.send(JSON.stringify({"type": "chat", "name": name, "msg": data()}))
        console.log("sent");
      }}>Send</button>
    </div>
  );
};

export default App;
