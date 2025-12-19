import Phaser from 'phaser';
import { StoryManager } from '../story/StoryManager';

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

    const story = StoryManager.getInstance();
    story.setEpisodeId('ep1');
    story.trigger('TRG_BOOT_000');
    story.trigger('TRG_BOOT_010');
    story.trigger('TRG_BOOT_020');
    story.trigger('TRG_BOOT_030');
    story.trigger('TRG_BOOT_040');

    // Set up global game data
    this.registry.set('power', 100);
    this.registry.set('data', 0);

    // Transition to preloader
    this.scene.start('PreloaderScene');
  }
}
