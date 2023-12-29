import { For, createSignal } from "solid-js";

function dayDisplay(cont: string) {
  let color: string;
  if (cont == "일") {
    color = "text-red-500"
  } else if (cont == "토") {
    color = "text-blue-500"
  } else {
    color = "text-black"
  }
  return (
  <div class="font-normal h-full flex" style="width: 14%">
    <div class={`mt-auto mb-auto text-center w-full font-semibold ${color}`}>{cont}</div>
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

function daySingle(num: number) {
  return (
    <div class="font-normal h-full flex" style="width: 14%">
      <div class='mt-auto mb-auto text-center w-full font-semibold'>{num}</div>
    </div>
  );
}

function day(date: Date) {
  const fy = date.getFullYear();
  const fm = date.getMonth();
  const _dateList: number[] = getDateList(date);
  const dateList = [];
  let i = 0;
  while (i <= _dateList.length) {
    const d = [];
    d.push(daySingle(_dateList[i]));
    let i2 = 1;
    while ((new Date(fy, fm, _dateList[i+i2])).getDay() != 0 && i+i2 < _dateList.length) {
      d.push(daySingle(_dateList[i+i2]));
      i2++;
      console.log(i+i2);
    }
    i += i2;
    console.log(d);
    dateList.push(d);
  }
  dateList[dateList.length-1].shift();
  dateList[dateList.length-1].push(daySingle(_dateList[_dateList.length-1]+1));
  return (
    <For each={dateList}>
      {(item) => {
        return (
          <div class="flex flex-row">
            <For each={item}>{(item) => item}</For>
          </div>
        );
      }}
    </For>
  );
}

function App() {
  const dates = ["일", "월", "화", "수", "목", "금", "토"]
  const [date, setDate] = createSignal(new Date());

  return (
    <div>
      <div class="bg-white w-screen h-screen flex flex-col">
        <div class="bg-gray-300 w-screen flex flex-col h-32">
          <div class="m-auto flex flex-row gap-2">
            <div class="text-4xl font-bold">2023/12</div>
          </div>
          <div class="flex flex-row w-full h-1/3">
            <For each={dates}>{(item) => dayDisplay(item)}</For>
          </div>
        </div>
        <div>
          {day(date())}
        </div>
      </div>
    </div>
  );
}

export default App;
