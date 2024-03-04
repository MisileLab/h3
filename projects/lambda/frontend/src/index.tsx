/* @refresh reload */
import './index.css';
import { render } from 'solid-js/web';
import { Route, Router } from '@solidjs/router';

import App from './App';
import AnA from './ana';
import Admin from './Admin';

const root = document.getElementById('root');

if (import.meta.env.DEV && !(root instanceof HTMLElement)) {
  throw new Error(
    'Root element not found. Did you forget to add it to your index.html? Or maybe the id attribute got misspelled?',
  );
}

render(() => 
  <Router>
    <Route path="/" component={App} />
    <Route path="/ana" component={AnA}/>
    <Route path="/admin" component={Admin}/>
  </Router>
, root!);
