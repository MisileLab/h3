<script>
    import { link } from 'svelte-spa-router';
</script>

<fieldset>
    <legend>Energy storage</legend>
    <p>storage: 1000/100000</p>
    <p>input: 0/100</p>
    <p>output: 0/100</p>
    unit:
    <select>
        <option id="MJ">Mekanism Joules (MJ)</option>
        <option id="J">Joules (J)</option>
        <option id="RF">Redstone Flux (RF)</option>
        <option id="EU">Energy Unit (EU)</option>
    </select>
</fieldset>

<fieldset>
    <legend>Electric Control</legend>
    <input type="checkbox" id="pneu" name="pneu">
    <label for="pneu">PneumaticCraft</label>
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
