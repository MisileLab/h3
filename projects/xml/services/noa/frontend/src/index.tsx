/* @refresh reload */
import './index.css';
import { render } from 'solid-js/web';

import App from './App';
import Upload from "./Upload";
import Gpg from "./Gpg";
import { MetaProvider } from '@solidjs/meta';
import { Route, Router } from '@solidjs/router';

const root = document.getElementById('root');

if (import.meta.env.DEV && !(root instanceof HTMLElement)) {
  throw new Error(
    'Root element not found. Did you forget to add it to your index.html? Or maybe the id attribute got misspelled?',
  );
}

render(() =>
<MetaProvider>
  <Router>
    <Route path="/upload" component={Upload} />
    <Route path="/gpg" component={Gpg} />
    <Route path="*path" component={App} />
  </Router>
</MetaProvider>
, root!);
