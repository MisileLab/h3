val iron = <tag:items:forge:ingots/iron>;
val redstone = <tag:items:forge:dusts/redstone>;
val casing = <tag:items:create:casing>;
val none = <item:minecraft:air>;

craftingTable.addShaped("physics_infuser", <item:vs_clockwork:physics_infuser>,[
  [
    iron,iron,iron
  ],[
    iron,casing,iron
  ],[
    iron,redstone,iron
  ]
]);

craftingTable.addShaped("phys_bearing", <item:vs_clockwork:phys_bearing>,[
  [
    none,iron,none
  ],[
    iron,<item:vs_clockwork:physics_infuser>,iron
  ],[
    none,<item:create:shaft>,none
  ]
]);
