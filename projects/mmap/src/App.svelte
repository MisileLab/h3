<script>
  import Router from 'svelte-spa-router';
  import Main from './routes/main.svelte';
  import Mek from './routes/mek.svelte';
  import Pneu from './routes/pneu.svelte';
  const routes = {
    '/': Main,
    '/mek': Mek,
    '/pneu': Pneu,
  }
</script>

<main>
  <Router {routes} />
</main>
