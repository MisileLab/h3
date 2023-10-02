<script>
    import { goto } from '$app/navigation';
    console.log(history.state);
    let pw = history.state["password"];
    if (pw == undefined) {
        goto("/")
    }
</script>

<h1>Welcome to password manager</h1>
<h2>Password is hidden</h2>
<p style="display: none;">password is {pw}</p>
<button type="button" on:click={()=>{goto("/3")}}>goto prev</button>
