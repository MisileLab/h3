import Fa from 'solid-fa';
import { faBars } from '@fortawesome/free-solid-svg-icons';

export default function App() {
  return (
    <div>
      <div class="bg-gray-800 w-1/4 h-screen font-mono text-white flex-wrap justify-start">
        <div class="ml-3 flex flex-col">
          <div class="h-16 flex">
            <div class="text-3xl mt-auto mb-auto flex justify-between w-full">
              <h1>Menu</h1>
              <Fa icon={faBars} class="mr-3" />
            </div>
          </div>
          <div>
            <h1>asd</h1>
          </div>
        </div>
      </div>
    </div>
  )
}