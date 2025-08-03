import Phaser from 'phaser';

import { MainMenuScene } from './game/scenes/MainMenuScene';
import { ExplorationScene } from './game/scenes/ExplorationScene';
import { NodeDetailScene } from './game/scenes/NodeDetailScene';

const gameConfig: Phaser.Types.Core.GameConfig = {
  type: Phaser.AUTO,
  width: 1200,
  height: 800,
  backgroundColor: '#1a1a2e',
  parent: 'app',
  scene: [MainMenuScene, ExplorationScene, NodeDetailScene],
  physics: {
    default: 'arcade',
    arcade: {
      gravity: { x: 0, y: 0 },
      debug: false
    }
  },
  scale: {
    mode: Phaser.Scale.FIT,
    autoCenter: Phaser.Scale.CENTER_BOTH,
    min: {
      width: 800,
      height: 600
    },
    max: {
      width: 1600,
      height: 1200
    }
  }
};

// Initialize game
const game = new Phaser.Game(gameConfig);

// Tauri integration for desktop features
if ((window as any).__TAURI__) {
  // TODO: Set window title when Tauri API is properly configured
  console.log('Tauri environment detected');
}

export default game;
