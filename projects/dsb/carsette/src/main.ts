import Phaser from 'phaser';
import { gameConfig } from './game/config';
import { BootScene } from './game/scenes/BootScene';
import { PreloaderScene } from './game/scenes/PreloaderScene';
import { TitleScene } from './game/scenes/TitleScene';
import { RunScene } from './game/scenes/RunScene';
import { CombatScene } from './game/scenes/CombatScene';
import { BlueprintScene } from './game/scenes/BlueprintScene';
import { ResultScene } from './game/scenes/ResultScene';
import { UIManager } from './ui/UIManager';

window.addEventListener('load', () => {
  // Initialize UI Manager
  const uiManager = UIManager.getInstance();
  uiManager.populateInventoryGrid();

  // Create Phaser game instance
  const config: Phaser.Types.Core.GameConfig = {
    ...gameConfig,
    scene: [BootScene, PreloaderScene, TitleScene, RunScene, CombatScene, BlueprintScene, ResultScene],
  };

  const game = new Phaser.Game(config);

  // Log game initialization
  console.log('CASSETTE TACTICS - System Online');
  console.log('Game initialized:', game);
});
