import Phaser from 'phaser';

export class PreloaderScene extends Phaser.Scene {
  private loadingText!: Phaser.GameObjects.Text;

  constructor() {
    super({ key: 'PreloaderScene' });
  }

  preload(): void {
    const centerX = this.cameras.main.centerX;
    const centerY = this.cameras.main.centerY;

    // Loading text
    this.loadingText = this.add.text(centerX, centerY, 'LOADING SYSTEM...', {
      fontFamily: 'VT323',
      fontSize: '32px',
      color: '#FFB000',
    });
    this.loadingText.setOrigin(0.5);

    // Placeholder: Load game assets here
    // this.load.image('player', 'assets/player.png');
    // this.load.image('enemy', 'assets/enemy.png');
    // this.load.audio('music', 'assets/music.mp3');
  }

  create(): void {
    console.log('PreloaderScene: Assets loaded');
    
    // Add a small delay for effect
    this.time.delayedCall(500, () => {
      this.scene.start('TitleScene');
    });
  }
}
