import { type PreinitializedWritableAtom } from "nanostores";

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

export enum DataType {
  blog = "blog",
  news = "news"
}

export enum AnimationValue {
  done = "done",
  doing = "doing"
}

function wrapFunction(f: any, connectedCallback: ()=>void) {
  return () => {try {
    f();
  } catch (e) {
    console.log("fallback")
    if (e instanceof indexError) {
      connectedCallback();
    } else {
      throw e;
    }
  }}
}

function safeCaptureStackTrace(err: Error, constructorOpt: unknown) {
  if ('captureStackTrace' in Error) {
    // @ts-ignore
    Error.captureStackTrace(err, constructorOpt);
  } else {
    // Fallback to manual stack trace construction
    // @ts-ignore
    const {stack} = new Error();
    err.stack = stack || '';
  }
}

export class indexError extends Error {
  constructor() {
    super("indexError");
    safeCaptureStackTrace(this, indexError);
  }
}

function replaceCharacterAtIndex(string: string, index: number, replacement: string) {
  // Check if the index is within bounds
  if (index < 0 || index >= string.length) {
    throw new indexError();
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
  connectedCallback: ()=>void,
  a: HTMLElement,
  complete: number,
  init: number,
  progress: PreinitializedWritableAtom<AnimationValue>,
  dontChange: boolean = false
) {
  const start = 0;
  const end = a.innerText.length;
  const ended = range(start, end);
  for (let i=start;i<end;i++) {
    setTimeout(wrapFunction(()=>{
      const _rand = Math.floor(Math.random() * ended.length);
      const rand = ended[_rand];
      ended.splice(_rand, 1)
      a.innerText=replaceCharacterAtIndex(a.innerText, rand, '_')
      if (!dontChange && i===end-1) {progress.set(AnimationValue.done);}
    }, connectedCallback), init+complete*i);
  }
}

export function normalHide(
  connectedCallback: ()=>void,
  a: HTMLElement,
  rewind: boolean,
  complete: number,
  init: number,
  progress: PreinitializedWritableAtom<AnimationValue>,
  dontChange: boolean = false
) {
  const l = a.innerText.length-1;
  for (let i=0;i<=l;i++) {
    setTimeout(wrapFunction(()=>{
      a.innerText=replaceCharacterAtIndex(a.innerText, rewind?l-i:i, '_')
      if (!dontChange && i===l) {progress.set(AnimationValue.done);}
    }, connectedCallback), init+complete*i);
  }
}

export function randomShow(
  connectedCallback: ()=>void,
  a: HTMLElement,
  original: string,
  complete: number,
  init: number,
  progress: PreinitializedWritableAtom<AnimationValue>,
  dontChange: boolean = false
) {
  const start = 0;
  const end = a.innerText.length;
  const ended = range(start, end);
  for (let i=start;i<end;i++) {
    setTimeout(wrapFunction(()=>{
      const _rand = Math.floor(Math.random() * ended.length);
      const rand = ended[_rand];
      ended.splice(_rand, 1)
      a.innerText=replaceCharacterAtIndex(a.innerText, rand, original[rand])
      if (!dontChange && i===end-1) {progress.set(AnimationValue.done);}
    }, connectedCallback), init+complete*i);
  }
}

export function normalShow(
  connectedCallback: ()=>void,
  a: HTMLElement,
  original: string,
  rewind: boolean,
  complete: number,
  init: number,
  progress: PreinitializedWritableAtom<AnimationValue>,
  dontChange: boolean = false
) {
  const l = a.innerText.length-1;
  for (let i=0;i<=l;i++) {
    setTimeout(wrapFunction(()=>{
      a.innerText=replaceCharacterAtIndex(a.innerText, rewind?l-i:i, original[rewind?l-i:i]);
      if (!dontChange && i===l) {progress.set(AnimationValue.done);}
    }, connectedCallback), init+complete*i);
  }
}

export function isRealInside(element: HTMLElement, event: MouseEvent) {
  const rect = element.getBoundingClientRect();
  return (
    event.clientX >= rect.left &&
    event.clientX <= rect.right &&
    event.clientY >= rect.top &&
    event.clientY <= rect.bottom
  );
}
