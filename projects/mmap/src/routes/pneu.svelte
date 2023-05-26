<script>
    import { link } from 'svelte-spa-router';
</script>

<fieldset>
    <legend>Pressure Storage</legend>
    <p>pressure: 5bar</p>
    <p>air: 0/100000</p>
    <p>input: 0/100</p>
    <p>output: 0/100</p>
</fieldset>

<a use:link href="/">go back</a>

<style>
    fieldset {
        border: 2px solid white;
    }
    p {
        padding: 0;
        margin: 0;
    }
</style>
