import { render } from 'solid-js/web';
import { Router, Routes, Route } from '@solidjs/router';
import Desktop4 from './pages/Desktop4';

render(
  () => (
    <Router>
      <Routes>
        <Route path="/" element={<Desktop4 />} />
      </Routes>
    </Router>
  ), document.getElementById('root')!!
)
