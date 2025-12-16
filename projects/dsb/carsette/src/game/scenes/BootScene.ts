import Phaser from 'phaser';

export class BootScene extends Phaser.Scene {
  constructor() {
    super({ key: 'BootScene' });
  }

  preload(): void {
    // Load any essential boot assets here
    // For now, we'll use placeholders
  }

  create(): void {
    console.log('BootScene: System initializing...');
    
    // Set up global game data
    this.registry.set('power', 100);
    this.registry.set('data', 0);
    
    // Transition to preloader
    this.scene.start('PreloaderScene');
  }
}
