import Phaser from 'phaser';
import { gameConfig } from './game/config';
import { BootScene } from './game/scenes/BootScene';
import { PreloaderScene } from './game/scenes/PreloaderScene';
import { TitleScene } from './game/scenes/TitleScene';
import { CampusScene } from './game/scenes/CampusScene';
import { BattleScene } from './game/scenes/BattleScene';
import { UIManager } from './ui/UIManager';

window.addEventListener('load', () => {
  // Initialize UI Manager
  const uiManager = UIManager.getInstance();
  uiManager.populateInventoryGrid();

  // Create Phaser game instance
  const config: Phaser.Types.Core.GameConfig = {
    ...gameConfig,
    scene: [BootScene, PreloaderScene, TitleScene, CampusScene, BattleScene],
  };

  const game = new Phaser.Game(config);

  // Log game initialization
  console.log('CASSETTE TACTICS - System Online');
  console.log('Game initialized:', game);
});
