<script>
import { goto } from "$app/navigation";
</script>
<style>
    .invisible {
        display: none;
    }
</style>
<h1>This is level 1</h1>
<input type="text" placeholder="input password" on:input={
    () => {if (event.target.value == "0xc0ffee") {goto("/2")}}
}/>
<h1 class="invisible">password is 0xc0ffee</h1>
