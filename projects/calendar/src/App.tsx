import { For, createSignal } from "solid-js";

function getColor(cont: string) {
  if (cont == "일") {
    return "text-red-500";
  } else if (cont == "토") {
    return "text-blue-500";
  } else {
    return "text-black";
  }
}

function dayDisplay(cont: string) {
  return (
  <div class="font-normal h-full flex" style="width: calc(100vw / 7);"> 
    <div class={`mt-auto mb-auto text-center w-full font-semibold ${getColor(cont)}`}>{cont}</div>
  </div>
  );
}

function getDateList(date: Date) {
  const lastDate = new Date(date.getFullYear(), date.getMonth(), 0).getDate();
  const dateList = [];
  for (let i = 1; i <= lastDate; i++) {
    dateList.push(i);
  }
  return dateList;
}

function daySingle(num: number | undefined = undefined, today: boolean = false) {
  let cont: string | number;
  if (num === undefined) {cont = "";} else {cont = num;}
  return (
    <div class={today ? "bg-gray-400" : ""} style='width: calc(100vw / 7)'>
      <div class='mb-auto text-right mr-4 mt-2 font-semibold'>{cont}</div>
    </div>
  );
}

function day(date: Date) {
  const fy = date.getFullYear();
  const fm = date.getMonth();
  const fd = date.getDate();
  const _dateList: number[] = getDateList(date);
  const dateList = [];
  let i = 0;
  while (i <= _dateList.length) {
    const d = [];
    d.push(daySingle(_dateList[i], _dateList[i] === fd));
    let i2 = 1;
    while ((new Date(fy, fm, _dateList[i+i2])).getDay() != 0 && i+i2 < _dateList.length) {
      d.push(daySingle(_dateList[i+i2], _dateList[i+i2] === fd));
      i2++;
    }
    if (d.length < 7 && i == 0) {
      let tmp = 0;
      const length = d.length;
      while (tmp <= 6-length) {
        d.unshift(daySingle());
        tmp++;
      }
    }
    i += i2;
    dateList.push(d);
  }
  dateList[dateList.length-1].shift();
  dateList[dateList.length-1].push(daySingle(_dateList[_dateList.length-1]+1));
  return (
    <For each={dateList}>
      {(item) => {
        return (
          <div class="flex flex-row h-1/5 border-gray-700 border-solid" style="border-top-width: 0.5px;">
            <For each={item}>{(item) => item}</For>
          </div>
        );
      }}
    </For>
  );
}

function App() {
  const dates = ["일", "월", "화", "수", "목", "금", "토"];
  const today = new Date();
  const [date, setDate] = createSignal(today);

  return (
    <div>
      <div class="bg-white w-screen h-screen flex flex-col">
        <div class="bg-gray-300 w-screen flex flex-col h-32">
          <div class="m-auto flex flex-row gap-2">
            <div class="text-4xl font-bold">{`${date().getFullYear()}/${date().getMonth()+1}`}</div>
          </div>
          <div class="flex flex-row w-full h-1/3">
            <For each={dates}>{(item) => {
              return dayDisplay(item);
            }}</For>
          </div>
        </div>
        <div class="h-full flex flex-col">
          {day(date())}
        </div>
      </div>
    </div>
  );
}

export default App;
