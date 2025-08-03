import Phaser from 'phaser';

export class MainMenuScene extends Phaser.Scene {

  constructor() {
    super({ key: 'MainMenuScene' });
  }

  create(): void {
    this.createBackground();
    this.createTitle();
    this.createMenuButtons();
    this.createParticles();
  }

  private createBackground(): void {
    // Create gradient background
    const graphics = this.add.graphics();
    graphics.fillGradientStyle(0x1a1a2e, 0x16213e, 0x0f3460, 0x533483, 1);
    graphics.fillRect(0, 0, this.scale.width, this.scale.height);

    // Add some decorative elements
    graphics.fillStyle(0xffffff, 0.1);
    for (let i = 0; i < 100; i++) {
      const x = Math.random() * this.scale.width;
      const y = Math.random() * this.scale.height;
      const size = Math.random() * 3 + 1;
      graphics.fillCircle(x, y, size);
    }
  }

  private createTitle(): void {
    const title = this.add.text(this.scale.width / 2, 150, 'NODE EXPLORER', {
      fontSize: '64px',
      fontFamily: 'Arial, sans-serif',
      color: '#ffffff',
      fontStyle: 'bold'
    });
    title.setOrigin(0.5);

    // TODO: Add glow effect when Phaser shadow API is available

    // Animate title
    this.tweens.add({
      targets: title,
      y: 160,
      duration: 2000,
      ease: 'Sine.inOut',
      yoyo: true,
      repeat: -1
    });

    const subtitle = this.add.text(this.scale.width / 2, 220, 'Explore the interconnected realms', {
      fontSize: '24px',
      fontFamily: 'Arial, sans-serif',
      color: '#b8c5d6'
    });
    subtitle.setOrigin(0.5);
  }

  private createMenuButtons(): void {
    const buttonStyle = {
      fontSize: '32px',
      fontFamily: 'Arial, sans-serif',
      color: '#ffffff',
      backgroundColor: '#4a90e2',
      padding: { x: 20, y: 10 }
    };

    const startButton = this.add.text(this.scale.width / 2, 350, 'START EXPLORATION', buttonStyle);
    startButton.setOrigin(0.5);
    startButton.setInteractive({ useHandCursor: true });

    const continueButton = this.add.text(this.scale.width / 2, 420, 'CONTINUE GAME', buttonStyle);
    continueButton.setOrigin(0.5);
    continueButton.setInteractive({ useHandCursor: true });

    const settingsButton = this.add.text(this.scale.width / 2, 490, 'SETTINGS', buttonStyle);
    settingsButton.setOrigin(0.5);
    settingsButton.setInteractive({ useHandCursor: true });

    // Button hover effects
    [startButton, continueButton, settingsButton].forEach(button => {
      button.on('pointerover', () => {
        button.setStyle({ ...buttonStyle, backgroundColor: '#357abd' });
        button.setScale(1.1);
      });

      button.on('pointerout', () => {
        button.setStyle(buttonStyle);
        button.setScale(1);
      });
    });

    // Button click handlers
    startButton.on('pointerdown', () => {
      this.scene.start('ExplorationScene');
    });

    continueButton.on('pointerdown', () => {
      // TODO: Implement save/load functionality
      this.scene.start('ExplorationScene');
    });

    settingsButton.on('pointerdown', () => {
      // TODO: Implement settings menu
      console.log('Settings clicked');
    });
  }

  private createParticles(): void {
    // TODO: Implement particle system when Phaser particles API is available
  }
} 