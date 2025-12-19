import Phaser from 'phaser';
import { ProgressManager } from '../systems/ProgressManager';
import { StoryManager } from '../story/StoryManager';
import { RunManager } from '../systems/RunManager';
import { UIManager } from '../../ui/UIManager';


export class ResultScene extends Phaser.Scene {
  private runManager!: RunManager;
  private uiManager!: UIManager;

  constructor() {
    super({ key: 'ResultScene' });
  }

  create(): void {
    this.runManager = RunManager.getInstance();
    this.uiManager = UIManager.getInstance();
    this.uiManager.updateHeat(this.runManager.getHeat());
    this.uiManager.updateStabilizer(this.runManager.getStabilizerCharges());

    const centerX = this.cameras.main.centerX;
    const centerY = this.cameras.main.centerY;

    const progressManager = ProgressManager.getInstance();
    const storyManager = StoryManager.getInstance();
    storyManager.setEpisodeId(this.runManager.getEpisode().id);

    const completed = this.runManager.isRunComplete();
    if (completed) {
      if (this.runManager.getEpisode().id === 'ep1') {
        storyManager.trigger('TRG_RUN_COMPLETE');
        const extractionNodeId = this.runManager.getLastExtractionNodeId();
        if (extractionNodeId === 'N3A') {
          storyManager.trigger('TRG_REWARD_SUMMARY_A');
        } else if (extractionNodeId === 'N3B') {
          storyManager.trigger('TRG_REWARD_SUMMARY_B');
        }
      }

      progressManager.markEpisodeComplete(this.runManager.getEpisode().id);
    }

    const title = completed ? 'RUN COMPLETE' : 'RUN INCOMPLETE';


    this.add.text(centerX, centerY - 40, title, {
      fontFamily: 'VT323',
      fontSize: '32px',
      color: '#FFB000',
    }).setOrigin(0.5);

    this.add.text(centerX, centerY + 10, `HEAT: ${this.runManager.getHeat()}`, {
      fontFamily: 'VT323',
      fontSize: '20px',
      color: '#B026FF',
    }).setOrigin(0.5);

    this.add.text(centerX, centerY + 40, 'Press SPACE to return to Title', {
      fontFamily: 'VT323',
      fontSize: '16px',
      color: '#FFFFFF',
    }).setOrigin(0.5);

    this.input.keyboard?.once('keydown-SPACE', () => {
      this.runManager.reset();
      this.scene.start('TitleScene');
    });
  }
}
