/* @refresh reload */
import './index.css';
import { render } from 'solid-js/web';

import App from './App';
import About from './About';
import { Route, Router } from '@solidjs/router';

const root = document.getElementById('root');

if (import.meta.env.DEV && !(root instanceof HTMLElement)) {
  throw new Error(
    'Root element not found. Did you forget to add it to your index.html? Or maybe the id attribute got misspelled?',
  );
}

render(() => 
  <Router>
    <Route path="/" component={App} />
    <Route path="/about" component={About} />
  </Router>
, root!);
