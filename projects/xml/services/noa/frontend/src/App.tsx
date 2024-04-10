import { Title } from "@solidjs/meta";

export default function App() {
  return (
    <div class="w-screen h-screen bg-ctp-crust">
      <Title>{window.location.pathname}</Title>
    </div>
  );
};

