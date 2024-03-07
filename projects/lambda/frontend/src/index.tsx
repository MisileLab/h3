/* @refresh reload */
import './index.css';
import { ErrorBoundary, render } from 'solid-js/web';
import { Route, Router } from '@solidjs/router';

import App from './App';
import AnA from './ana';
import Admin from './Admin';
import Login from './login';

const root = document.getElementById('root');

if (import.meta.env.DEV && !(root instanceof HTMLElement)) {
  throw new Error(
    'Root element not found. Did you forget to add it to your index.html? Or maybe the id attribute got misspelled?',
  );
}

render(() => 
  <ErrorBoundary fallback={(err) => {console.error(err);return (<div class="font-bold flex justify-center items-center w-screen h-screen text-4xl text-center">
    01043782246으로 에러와 함께 검토 후 문자 주시면 매점 사드림
    <br />
    :sunglasses:
  </div>)}}>
    <Router>
      <Route path="/" component={App} />
      <Route path="/ana" component={AnA}/>
      <Route path="/admin" component={Admin}/>
      <Route path="/login" component={Login}/>
    </Router>
  </ErrorBoundary>
, root!);
