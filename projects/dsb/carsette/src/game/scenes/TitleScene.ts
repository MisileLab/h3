import Phaser from 'phaser';
import { ProgressManager } from '../systems/ProgressManager';
import { StoryManager } from '../story/StoryManager';
import { RunManager } from '../systems/RunManager';

export class TitleScene extends Phaser.Scene {
  private titleText!: Phaser.GameObjects.Text;
  private startText!: Phaser.GameObjects.Text;
  private asciiLogo!: Phaser.GameObjects.Text;
  private runManager!: RunManager;
  private progressManager!: ProgressManager;


  constructor() {
    super({ key: 'TitleScene' });
  }

  create(): void {
    this.runManager = RunManager.getInstance();
    this.progressManager = ProgressManager.getInstance();
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
    this.startText = this.add.text(centerX, centerY + 150, this.getProgressLine(), {
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

    const buttonY = centerY + 190;

    const startButton = this.add.text(centerX - 80, buttonY, 'START', {
      fontFamily: 'VT323',
      fontSize: '18px',
      color: '#FFB000',
      backgroundColor: '#3d2a00',
      padding: { left: 12, right: 12, top: 6, bottom: 6 },
    }).setOrigin(0.5);

    const resetButton = this.add.text(centerX + 80, buttonY, 'RESET', {
      fontFamily: 'VT323',
      fontSize: '18px',
      color: '#FFB000',
      backgroundColor: '#3d2a00',
      padding: { left: 12, right: 12, top: 6, bottom: 6 },
    }).setOrigin(0.5);

    startButton.setInteractive({ useHandCursor: true });
    resetButton.setInteractive({ useHandCursor: true });

    startButton.on('pointerdown', () => this.startNextEpisode());
    resetButton.on('pointerdown', () => this.resetProgress());

  }

  private startNextEpisode(): void {
    const nextEpisodeId = this.progressManager.getNextEpisodeId();
    this.startEpisode(nextEpisodeId);
  }

  private resetProgress(): void {
    this.progressManager.resetProgress();

    const story = StoryManager.getInstance();
    story.setEpisodeId('ep1');
    story.resetRun();

    this.runManager.setEpisode('ep1');
    this.startText.setText(this.getProgressLine());
  }

  private getProgressLine(): string {
    const completed = this.progressManager.getCompletedEpisodes();
    if (completed.length >= 3) return 'ALL EPISODES COMPLETE // REPLAY EP3';
    return `NEXT EPISODE: ${this.progressManager.getNextEpisodeId().toUpperCase()}`;
  }

  private startEpisode(episodeId: string): void {
    this.triggerGlitchEffect();

    const story = StoryManager.getInstance();
    story.setEpisodeId(episodeId);
    story.resetRun();

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
