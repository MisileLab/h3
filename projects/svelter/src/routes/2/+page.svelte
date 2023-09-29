<script>
import { goto } from '$app/navigation';
</script>
<style>
    .password{passwordis: A_SIMPLE_FLAG}
</style>
<h1 class="password">This is Level 2</h1>
<input type="text" placeholder="input text" on:input={()=>{
    if(event.target.value=="A_SIMPLE_FLAG"){
        goto("/3")
    }
}}/>
