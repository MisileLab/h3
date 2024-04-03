/* @refresh reload */
import './index.css';
import { render } from 'solid-js/web';
import { MetaProvider } from '@solidjs/meta';
import { Route, Router } from '@solidjs/router';

import Login from './Login';
import Register from "./Register";

const root = document.getElementById('root');

if (import.meta.env.DEV && !(root instanceof HTMLElement)) {
  throw new Error(
    'Root element not found. Did you forget to add it to your index.html? Or maybe the id attribute got misspelled?',
  );
}

render(() => 
  <MetaProvider>
    <Router>
      <Route path="/" component={Login} />
      <Route path="/register" component={Register} />
    </Router>
  </MetaProvider>
, root!);
