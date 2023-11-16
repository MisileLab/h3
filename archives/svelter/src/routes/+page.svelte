<script>
import { goto } from "$app/navigation";
</script>
<h1>This is Level 0, password is this level number</h1>
<input type="text" placeholder="password input" on:input={() => {
    if (event.target.value == "0") {
        goto('/1')
    }
}}/>
