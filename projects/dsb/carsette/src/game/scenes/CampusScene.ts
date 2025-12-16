import Phaser from 'phaser';
import { UIManager } from '../../ui/UIManager';

export class CampusScene extends Phaser.Scene {
  private graphics!: Phaser.GameObjects.Graphics;
  private titleText!: Phaser.GameObjects.Text;
  private buildings: { name: string; x: number; y: number }[] = [];
  private uiManager!: UIManager;

  constructor() {
    super({ key: 'CampusScene' });
  }

  create(): void {
    this.uiManager = UIManager.getInstance();
    this.uiManager.updateSystemMessage('CAMPUS MAP ONLINE');

    const centerX = this.cameras.main.centerX;
    const centerY = this.cameras.main.centerY;

    // Title
    this.titleText = this.add.text(centerX, 30, 'CAMPUS // BLUEPRINT VIEW', {
      fontFamily: 'VT323',
      fontSize: '28px',
      color: '#FFB000',
    });
    this.titleText.setOrigin(0.5);

    // Draw grid background
    this.drawGrid();

    // Create building nodes
    this.buildings = [
      { name: 'DORM', x: centerX - 150, y: centerY - 80 },
      { name: 'FACTORY', x: centerX + 150, y: centerY - 80 },
      { name: 'LAB', x: centerX - 150, y: centerY + 80 },
      { name: 'HANGAR', x: centerX + 150, y: centerY + 80 },
    ];

    this.buildings.forEach((building) => {
      this.drawBuilding(building.name, building.x, building.y);
    });

    // Instructions
    const instructionText = this.add.text(
      centerX,
      this.cameras.main.height - 40,
      'CLICK ANY BUILDING TO DEPLOY',
      {
        fontFamily: 'VT323',
        fontSize: '20px',
        color: '#594100',
      }
    );
    instructionText.setOrigin(0.5);

    this.tweens.add({
      targets: instructionText,
      alpha: 0.3,
      duration: 1000,
      yoyo: true,
      repeat: -1,
    });
  }

  private drawGrid(): void {
    this.graphics = this.add.graphics();
    this.graphics.lineStyle(1, 0x594100, 0.3);

    const width = this.cameras.main.width;
    const height = this.cameras.main.height;
    const gridSize = 40;

    // Vertical lines
    for (let x = 0; x < width; x += gridSize) {
      this.graphics.lineBetween(x, 0, x, height);
    }

    // Horizontal lines
    for (let y = 0; y < height; y += gridSize) {
      this.graphics.lineBetween(0, y, width, y);
    }
  }

  private drawBuilding(name: string, x: number, y: number): void {
    const graphics = this.add.graphics();
    
    // Building rectangle
    graphics.lineStyle(3, 0xffb000, 1);
    graphics.strokeRect(-40, -30, 80, 60);
    
    // Fill with dim color
    graphics.fillStyle(0x594100, 0.2);
    graphics.fillRect(-40, -30, 80, 60);

    // Cross lines (blueprint style)
    graphics.lineStyle(1, 0xffb000, 0.5);
    graphics.lineBetween(-40, -30, 40, 30);
    graphics.lineBetween(40, -30, -40, 30);

    graphics.setPosition(x, y);

    // Building name
    const text = this.add.text(x, y, name, {
      fontFamily: 'VT323',
      fontSize: '18px',
      color: '#FFB000',
      align: 'center',
    });
    text.setOrigin(0.5);

    // Make interactive
    const hitZone = this.add.rectangle(x, y, 80, 60, 0xffffff, 0);
    hitZone.setInteractive({ useHandCursor: true });

    hitZone.on('pointerover', () => {
      graphics.clear();
      graphics.lineStyle(3, 0xb026ff, 1);
      graphics.strokeRect(-40, -30, 80, 60);
      graphics.fillStyle(0xb026ff, 0.2);
      graphics.fillRect(-40, -30, 80, 60);
      text.setColor('#B026FF');
    });

    hitZone.on('pointerout', () => {
      graphics.clear();
      graphics.lineStyle(3, 0xffb000, 1);
      graphics.strokeRect(-40, -30, 80, 60);
      graphics.fillStyle(0x594100, 0.2);
      graphics.fillRect(-40, -30, 80, 60);
      graphics.lineStyle(1, 0xffb000, 0.5);
      graphics.lineBetween(-40, -30, 40, 30);
      graphics.lineBetween(40, -30, -40, 30);
      text.setColor('#FFB000');
    });

    hitZone.on('pointerdown', () => {
      console.log(`Deploying to: ${name}`);
      this.cameras.main.flash(200, 255, 176, 0);
      
      this.time.delayedCall(300, () => {
        this.scene.start('BattleScene');
      });
    });
  }
}
