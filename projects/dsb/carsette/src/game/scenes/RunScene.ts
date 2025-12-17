import Phaser from 'phaser';
import { RunManager } from '../systems/RunManager';
import { NodeConfig } from '../types/Run';
import { UIManager } from '../../ui/UIManager';

export class RunScene extends Phaser.Scene {
  private runManager!: RunManager;
  private uiManager!: UIManager;
  private nodeTexts: Phaser.GameObjects.Text[] = [];

  constructor() {
    super({ key: 'RunScene' });
  }

  create(): void {
    this.runManager = RunManager.getInstance();
    this.uiManager = UIManager.getInstance();
    this.uiManager.updateSystemMessage(`${this.runManager.getEpisode().name} // RUN MAP`);
    this.uiManager.updateHeat(this.runManager.getHeat());
    this.uiManager.updateStabilizer(this.runManager.getStabilizerCharges());

    const episode = this.runManager.getEpisode();
    const centerX = this.cameras.main.centerX;
    const startY = 80;

    this.add.text(centerX, 30, `EPISODE: ${episode.name}`, {
      fontFamily: 'VT323',
      fontSize: '28px',
      color: '#FFB000',
    }).setOrigin(0.5);

    episode.nodes.forEach((node, index) => {
      const y = startY + index * 60;
      this.drawNode(node, centerX, y, index === this.runManager.getEpisode().nodes.indexOf(this.runManager.getCurrentNode()));
    });

    this.renderToneChoices(centerX, startY + episode.nodes.length * 60 + 30);
    this.renderExtractionBranches(centerX, startY + episode.nodes.length * 60 + 70);

    this.input.keyboard?.on('keydown-SPACE', () => {
      this.enterCurrentNode();
    });
  }

  private drawNode(node: NodeConfig, x: number, y: number, isActive: boolean): void {
    const box = this.add.rectangle(x, y, 520, 44, isActive ? 0xffb000 : 0x594100, 0.2);
    box.setStrokeStyle(2, isActive ? 0xffb000 : 0x594100, 0.8);

    const text = this.add.text(x - 240, y - 12, `${node.id}`, {
      fontFamily: 'VT323',
      fontSize: '18px',
      color: '#FFB000',
    });
    this.nodeTexts.push(text);

    this.add.text(x - 180, y - 12, node.name, {
      fontFamily: 'VT323',
      fontSize: '18px',
      color: '#FFB000',
    });

    this.add.text(x - 180, y + 6, node.description, {
      fontFamily: 'VT323',
      fontSize: '14px',
      color: '#B026FF',
    });

    const enterText = this.add.text(x + 200, y - 6, 'ENTER', {
      fontFamily: 'VT323',
      fontSize: '16px',
      color: '#FFFFFF',
      backgroundColor: isActive ? '#B026FF' : '#3d2a00',
      padding: { left: 8, right: 8, top: 4, bottom: 4 },
    });
    enterText.setInteractive({ useHandCursor: true });
    enterText.on('pointerdown', () => this.enterCurrentNode());
  }

  private renderToneChoices(x: number, y: number): void {
    const tone = this.runManager.getToneFlag();
    this.add.text(x, y - 16, 'DIALOG TONE', {
      fontFamily: 'VT323',
      fontSize: '14px',
      color: '#FFB000',
    }).setOrigin(0.5);

    const choices: { label: string; value: string }[] = [
      { label: 'EMPATHY', value: 'TONE_EMPATHY' },
      { label: 'PRAGMATIC', value: 'TONE_PRAGMATIC' },
      { label: 'CURIOUS', value: 'TONE_CURIOUS' },
    ];

    const offset = -120;
    choices.forEach((choice, index) => {
      const button = this.add.text(x + offset + index * 120, y, choice.label, {
        fontFamily: 'VT323',
        fontSize: '14px',
        color: '#FFFFFF',
        backgroundColor: tone === choice.value ? '#B026FF' : '#3d2a00',
        padding: { left: 8, right: 8, top: 4, bottom: 4 },
      }).setOrigin(0.5);
      button.setInteractive({ useHandCursor: true });
      button.on('pointerdown', () => {
        this.runManager.setToneFlag(choice.value);
        this.uiManager.updateSystemMessage(`TONE SET: ${choice.label}`);
        this.scene.restart();
      });
    });
  }

  private renderExtractionBranches(x: number, y: number): void {
    const hasBranchA = this.runManager.getEpisode().nodes.some(n => n.id === 'N3A' || n.id === 'N4A');
    const hasBranchB = this.runManager.getEpisode().nodes.some(n => n.id === 'N3B' || n.id === 'N4B');
    if (!hasBranchA || !hasBranchB) return;

    const buttonA = this.add.text(x - 80, y, 'ROUTE A', {
      fontFamily: 'VT323',
      fontSize: '14px',
      color: '#FFFFFF',
      backgroundColor: '#3d2a00',
      padding: { left: 8, right: 8, top: 4, bottom: 4 },
    }).setOrigin(0.5);

    const buttonB = this.add.text(x + 80, y, 'ROUTE B', {
      fontFamily: 'VT323',
      fontSize: '14px',
      color: '#FFFFFF',
      backgroundColor: '#3d2a00',
      padding: { left: 8, right: 8, top: 4, bottom: 4 },
    }).setOrigin(0.5);

    buttonA.setInteractive({ useHandCursor: true });
    buttonB.setInteractive({ useHandCursor: true });

    buttonA.on('pointerdown', () => {
      if (this.runManager.getEpisode().nodes.some(n => n.id === 'N3A')) {
        this.runManager.setCurrentNodeById('N3A');
      } else {
        this.runManager.setCurrentNodeById('N4A');
      }
      this.uiManager.updateSystemMessage('ROUTE A SELECTED');
      this.scene.restart();
    });

    buttonB.on('pointerdown', () => {
      if (this.runManager.getEpisode().nodes.some(n => n.id === 'N3B')) {
        this.runManager.setCurrentNodeById('N3B');
      } else {
        this.runManager.setCurrentNodeById('N4B');
      }
      this.uiManager.updateSystemMessage('ROUTE B SELECTED');
      this.scene.restart();
    });
  }

  private enterCurrentNode(): void {
    const node = this.runManager.getCurrentNode();
    if (node.type === 'combat') {
      this.scene.start('CombatScene', { nodeId: node.id });
    } else if (node.type === 'blueprint') {
      this.scene.start('BlueprintScene', { nodeId: node.id });
    } else {
      this.scene.start('ResultScene');
    }
  }
}
