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
    this.uiManager.updateSystemMessage('EP1 // OUTSKIRTS RUN MAP');
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
