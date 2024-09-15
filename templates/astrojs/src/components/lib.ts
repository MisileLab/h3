export enum HoverType {
  none = "none",
  hide = "hide",
  show = "show",
  reanimate = "reanimate"
}

export enum AnimationType {
  rewind = "rewind",
  normal = "normal",
  random = "random"
}

function replaceCharacterAtIndex(string: string, index: number, replacement: string) {
  // Check if the index is within bounds
  if (index < 0 || index >= string.length) {
    throw new Error('Index out of bounds');
  }

  // Extract the substring before the index
  const before = string.slice(0, index);
  
  // Extract the substring after the index
  const after = string.slice(index + 1);

  // Return the new string with the replacement character inserted
  return before + replacement + after;
}

function range(start: number, end: number, step = 1) {
  // Initialize an empty array to hold the range of numbers
  const result: number[] = [];
  
  // Check if step is valid to prevent an infinite loop
  if (step === 0) {
    throw new Error("Step cannot be zero.");
  }
  
  // Generate the range of numbers based on start, end, and step
  if (step > 0) {
    for (let i = start; i < end; i += step) {
      result.push(i);
    }
  } else {
    for (let i = start; i > end; i += step) {
      result.push(i);
    }
  }
  
  return result;
}

export function getLength(a: Record<any, any>) {
  return Object.keys(a).length;
}

export function randomHide(
  a: HTMLElement,
  index: number,
  complete: number,
  init: number,
  animations: Record<number, {show: Record<number, number>, hide: Record<number, number>}>
) {
  const start = 0;
  const end = a.innerText.length;
  const ended = range(start, end);
  for (let i=start;i<end;i++) {
    animations[index].hide[i] = setTimeout(()=>{
      const _rand = Math.floor(Math.random() * ended.length);
      const rand = ended[_rand];
      ended.splice(_rand, 1)
      a.innerText=replaceCharacterAtIndex(a.innerText, rand, '_')
      delete animations[index].hide[i];
    }, init+complete*i);
  }
}

export function normalHide(
  a: HTMLElement,
  index: number,
  rewind: boolean,
  complete: number,
  init: number,
  animations: Record<number, {show: Record<number, number>, hide: Record<number, number>}>
) {
  const l = a.innerText.length-1;
  for (let i=0;i<a.innerText.length;i++) {
    animations[index].hide[i] = setTimeout(()=>{
      a.innerText=replaceCharacterAtIndex(a.innerText, rewind?l-i:i, '_')
      delete animations[index].hide[i];
    }, init+complete*i);
  }
}

export function randomShow(
  a: HTMLElement,
  index: number,
  original: string,
  complete: number,
  init: number,
  animations: Record<number, {show: Record<number, number>, hide: Record<number, number>}>
) {
  const start = 0;
  const end = a.innerText.length;
  const ended = range(start, end);
  for (let i=start;i<end;i++) {
    animations[index].show[i] = setTimeout(()=>{
      const _rand = Math.floor(Math.random() * ended.length);
      const rand = ended[_rand];
      ended.splice(_rand, 1)
      a.innerText=replaceCharacterAtIndex(a.innerText, rand, original[rand])
      delete animations[index].show[i];
    }, init+complete*i);
  }
}

export function normalShow(
  a: HTMLElement,
  index: number,
  original: string,
  rewind: boolean,
  complete: number,
  init: number,
  animations: Record<number, {show: Record<number, number>, hide: Record<number, number>}>
) {
  const l = a.innerText.length-1;
  for (let i=0;i<a.innerText.length;i++) {
    animations[index].show[i] = setTimeout(()=>{
      a.innerText=replaceCharacterAtIndex(a.innerText, rewind?l-i:i, original[rewind?l-i:i]);
      delete animations[index].show[i];
    }, init+complete*i);
  }
}
