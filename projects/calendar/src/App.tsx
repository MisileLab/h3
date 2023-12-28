function temp() {
  return (
  <div class="font-normal h-full flex" style="width: 14%">
    <div class="mt-auto mb-auto text-center w-full font-semibold">일</div>
  </div>
  );
}

function App() {
  return (
    <div>
      <div class="bg-white w-screen h-screen">
        <div class="bg-gray-300 w-screen flex flex-col h-32">
          <div class="m-auto flex flex-row gap-2">
            <div class="text-4xl font-bold">2023/12/28</div>
            <div class="text-2xl font-normal mt-auto mb-auto">목요일</div>
          </div>
          <div class="flex flex-row w-full h-1/3">
            {temp()}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
