import Phaser from 'phaser';

export class TitleScene extends Phaser.Scene {
  private titleText!: Phaser.GameObjects.Text;
  private startText!: Phaser.GameObjects.Text;
  private asciiLogo!: Phaser.GameObjects.Text;

  constructor() {
    super({ key: 'TitleScene' });
  }

  create(): void {
    const centerX = this.cameras.main.centerX;
    const centerY = this.cameras.main.centerY;

    // ASCII Logo
    const logo = `
  ██████╗ █████╗ ███████╗███████╗███████╗████████╗████████╗███████╗
 ██╔════╝██╔══██╗██╔════╝██╔════╝██╔════╝╚══██╔══╝╚══██╔══╝██╔════╝
 ██║     ███████║███████╗███████╗█████╗     ██║      ██║   █████╗  
 ██║     ██╔══██║╚════██║╚════██║██╔══╝     ██║      ██║   ██╔══╝  
 ╚██████╗██║  ██║███████║███████║███████╗   ██║      ██║   ███████╗
  ╚═════╝╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝   ╚═╝      ╚═╝   ╚══════╝
                                                                     
         ████████╗ █████╗  ██████╗████████╗██╗ ██████╗███████╗     
         ╚══██╔══╝██╔══██╗██╔════╝╚══██╔══╝██║██╔════╝██╔════╝     
            ██║   ███████║██║        ██║   ██║██║     ███████╗     
            ██║   ██╔══██║██║        ██║   ██║██║     ╚════██║     
            ██║   ██║  ██║╚██████╗   ██║   ██║╚██████╗███████║     
            ╚═╝   ╚═╝  ╚═╝ ╚═════╝   ╚═╝   ╚═╝ ╚═════╝╚══════╝     
    `;

    this.asciiLogo = this.add.text(centerX, centerY - 100, logo, {
      fontFamily: 'Share Tech Mono',
      fontSize: '10px',
      color: '#FFB000',
      align: 'center',
    });
    this.asciiLogo.setOrigin(0.5);

    // Title text
    this.titleText = this.add.text(centerX, centerY + 100, 'SYSTEM BOOT SEQUENCE', {
      fontFamily: 'VT323',
      fontSize: '24px',
      color: '#FFB000',
    });
    this.titleText.setOrigin(0.5);

    // Start instruction
    this.startText = this.add.text(centerX, centerY + 150, 'CLICK TO INITIALIZE', {
      fontFamily: 'VT323',
      fontSize: '20px',
      color: '#594100',
    });
    this.startText.setOrigin(0.5);

    // Blinking effect
    this.tweens.add({
      targets: this.startText,
      alpha: 0.3,
      duration: 800,
      yoyo: true,
      repeat: -1,
    });

    // Click to start with glitch effect
    this.input.once('pointerdown', () => {
      this.triggerGlitchEffect();

      this.time.delayedCall(600, () => {
        this.scene.start('RunScene');
      });
    });
  }

  private triggerGlitchEffect(): void {
    // Add glitch class to container
    const container = document.getElementById('game-container');
    if (container) {
      container.classList.add('glitch-active');
      
      setTimeout(() => {
        container.classList.remove('glitch-active');
      }, 500);
    }

    // In-scene glitch effect
    this.cameras.main.shake(300, 0.01);
    
    // Flash effect
    this.cameras.main.flash(200, 176, 38, 255, true);
  }
}
