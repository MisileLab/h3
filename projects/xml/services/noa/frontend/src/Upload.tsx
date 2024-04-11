import { Title } from "@solidjs/meta";
import { A } from "@solidjs/router";

export default function Upload() {
  return (
    <div class="w-screen h-screen bg-ctp-crust flex justify-center items-center">
      <Title>{window.location.pathname}asd</Title>
      <div class="w-fit h-fit p-6 bg-ctp-overlay0">
        <table>
          <thead>
            <tr class="flex flex-row gap-4 text-ctp-text">
              <th>Name</th>
              <th>Size (Bytes)</th>
            </tr>
          <tr class="flex flex-row gap-4 text-ctp-subtext0">
            <A href="/a.txt"><th class="text-ctp-sky">a.txt</th></A>
            <th>1000000</th>
          </tr>
          </thead>
        </table>
      </div>
    </div>
  );
};

