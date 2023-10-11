import { createSignal, type Component, createEffect } from 'solid-js';

const ionps = [
  ['H', 'Na', 'K', 'Ag', 'NH₄'],
  ['Mg', 'Pb', 'Cu', 'Ca', 'Ba', 'Zn', 'Fe(II)'],
  ['Al', 'Fe(III)'],
]

const ionms = [
  ['Cl', 'OH', 'NO₃', 'I', 'MnO₄', 'CH₃COO'],
  ['S', 'SO₄', 'O', 'CO₃'],
  ['PO₄']
]

const getRandomElement = <T extends any[]>(arr: T): T[number] => {
  return arr[Math.floor(Math.random() * arr.length)];
};

const PorM=()=>{return Math.random()<0.5?ionps:ionms;}
const _getAnswer = (s: string) => {
  for (const [index, ion] of ionps.entries()) {
    if (ion.includes(s)) {
      return index + 1;
    }
  }
  for (const [index, ion] of ionms.entries()) {
    if (ion.includes(s)) {
      return -(index + 1);
    }
  }
}

const App: Component = () => {
  let point = 0;
  let hpoint = 0;
  const [getIon, setIon] = createSignal(getRandomElement(getRandomElement(PorM())));
  const [getValue, setValue] = createSignal("");
  const [getAnswer, setAnswer] = createSignal(_getAnswer(getIon()));

  createEffect(()=>{
    if(Number(getValue())==getAnswer()){
      point++;
      setIon(getRandomElement(getRandomElement(PorM())));
      setValue("");
      setAnswer(_getAnswer(getIon()));
    } else if(point > 0 && getValue() != "") {
      alert(`failed, point is ${point}, high point is ${hpoint}, answer is ${getAnswer()}`)
      if (point > hpoint) {hpoint = point}
      point = 0;
    }
  })

  return (
    <div>
      <h1>{getIon()}</h1>
      <input onChange={(e)=>setValue(e.target.value)} value={getValue()}/>
    </div>
  );
};

export default App;
