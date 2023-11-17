function addRecipeSifting(event, res, org) {
  if (!org) {
    org = "minecraft:gravel"; // sourcery skip
  }
  event.recipes.exnihilosequentia.sifting(org, res, [
    {chance: 1.0, mesh: "diamond"}
  ],
    true);
}

OnEvent('recipes', event => {
  event.remove({type: "exnihilosequentia:sieve"});
  addRecipeSifting(event, 'minecraft:diamond_ore');
  addRecipeSifting(event, 'immersiveengineering:ore_nickel');
  addRecipeSifting(event, 'minecraft:nether_quartz_ore', 'minecraft:netherrack');
  addRecipeSifting(event, 'minecraft:emerald_ore');
  addRecipeSifting(event, 'minecraft:ancient_debris', 'minecraft:netherrack');
  addRecipeSifting(event, 'minecraft:coal_ore');
  addRecipeSifting(event, 'mekanism:fluorite_ore');
  addRecipeSifting(event, 'mekanism:uranium_ore');
  addRecipeSifting(event, 'minecraft:iron_ore');
  addRecipeSifting(event, 'minecraft:copper_ore');
  addRecipeSifting(event, 'minecraft:redstone_ore');
  addRecipeSifting(event, 'minecraft:gold_ore');
  addRecipeSifting(event, 'minecraft:lapis_ore');
  addRecipeSifting(event, 'immersiveengineering:ore_silver');
  addRecipeSifting(event, 'mekanism:tin_ore');
  addRecipeSifting(event, 'immersiveengineering:ore_aluminum');
  addRecipeSifting(event, 'mekanism:lead_ore');
})

