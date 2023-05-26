<script>
  import { link } from 'svelte-spa-router';
</script>

<main class="container" style="padding-top: 0vh">
  <h1>Modded Minecraft Administrator Panel</h1>
  <p>Made by Misile</p>

  <div id="main_select" class="row">
  <a use:link href="/mek"><img class="col-md-offset-2" id="mek" src="steel_ingot.png" alt="steel" style="margin-right: 20px"></a>
  <a use:link href="/pneu"><img id="pneu" src="pneumatic.png" alt="pneumatic" style="margin-left: 20px" height="150"></a>
  </div>
</main>

<style>
  #main_select #mek:hover{
    filter: drop-shadow(0 0 1em #747bff);
    will-change: filter;
    transition: filter .2s ease-out;
  }

  #main_select #pneu:hover {
    filter: drop-shadow(0 0 1em #747bff);
    will-change: filter;
    transition: filter .2s ease-out;
  }
</style>
