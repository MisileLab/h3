import { createSignal, type Component } from 'solid-js';
import { ColorModeProvider, ColorModeScript, useColorMode} from '@kobalte/core';
import NavBar from './components/navbar';
import { Card, CardContent, CardHeader } from './components/ui/card';
import { Input } from './components/ui/input';
import { Button } from './components/ui/button';
import { setCookie } from './definition';

const Login: Component = () => {
  const [key, setKey] = createSignal("");
  return (
    <div>
      <ColorModeScript />
      <ColorModeProvider>
        <div class="flex flex-col h-screen">
          <NavBar />
          <div class="flex justify-center items-center flex-grow">
            <Card class="flex flex-col items-center w-fit">
              <CardHeader>
                <h1 class="font-bold text-2xl">어드민 키 입력</h1>
              </CardHeader>
              <CardContent>
                <div class="flex flex-col items-center gap-4">
                  <Input type="text" placeholder="key" value={key()} onChange={(self)=>setKey(self.target.value)} />
                  <Button onclick={()=>{
                    setCookie('key', key(), 1)
                  }}>확인</Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </ColorModeProvider>
    </div>
  );
};

export default Login;
