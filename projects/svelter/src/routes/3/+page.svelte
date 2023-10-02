<script>
import { goto } from '$app/navigation';
</script>

<h1>This is Level 3</h1>
<input type="text" placeholder="input password" on:input={()=>{
    if (event.target.value == 'starlighter') {
        goto('/4');
    }
}}>
<button style="display: none" on:click={()=>{
    goto('/l3', {state: {password: 'starlighter'}})
}}>try</button>
