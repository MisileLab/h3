import Phaser from 'phaser';
import { RunManager } from '../systems/RunManager';

export class TitleScene extends Phaser.Scene {
  private titleText!: Phaser.GameObjects.Text;
  private startText!: Phaser.GameObjects.Text;
  private asciiLogo!: Phaser.GameObjects.Text;
  private runManager!: RunManager;

  constructor() {
    super({ key: 'TitleScene' });
  }

  create(): void {
    this.runManager = RunManager.getInstance();
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

    const startEp1 = this.add.text(centerX - 140, centerY + 190, 'START EP1', {
      fontFamily: 'VT323',
      fontSize: '18px',
      color: '#FFB000',
      backgroundColor: '#3d2a00',
      padding: { left: 8, right: 8, top: 6, bottom: 6 },
    }).setOrigin(0.5);

    const startEp2 = this.add.text(centerX, centerY + 190, 'START EP2', {
      fontFamily: 'VT323',
      fontSize: '18px',
      color: '#FFB000',
      backgroundColor: '#3d2a00',
      padding: { left: 8, right: 8, top: 6, bottom: 6 },
    }).setOrigin(0.5);

    const startEp3 = this.add.text(centerX + 140, centerY + 190, 'START EP3', {
      fontFamily: 'VT323',
      fontSize: '18px',
      color: '#FFB000',
      backgroundColor: '#3d2a00',
      padding: { left: 8, right: 8, top: 6, bottom: 6 },
    }).setOrigin(0.5);

    startEp1.setInteractive({ useHandCursor: true });
    startEp2.setInteractive({ useHandCursor: true });
    startEp3.setInteractive({ useHandCursor: true });

    startEp1.on('pointerdown', () => this.startEpisode('ep1'));
    startEp2.on('pointerdown', () => this.startEpisode('ep2'));
    startEp3.on('pointerdown', () => this.startEpisode('ep3'));
  }

  private startEpisode(episodeId: string): void {
    this.triggerGlitchEffect();
    this.runManager.setEpisode(episodeId);
    this.time.delayedCall(600, () => {
      this.scene.start('RunScene');
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
