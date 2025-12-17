import Phaser from 'phaser';
import { InventoryManager } from '../systems/InventoryManager';
import { RunManager } from '../systems/RunManager';
import { UIManager } from '../../ui/UIManager';
import { ItemData, ItemType } from '../types/Item';

export class BlueprintScene extends Phaser.Scene {
  private inventoryManager!: InventoryManager;
  private runManager!: RunManager;
  private uiManager!: UIManager;
  private extractorBuilt: boolean = false;
  private refinerBuilt: boolean = false;
  private pressBuilt: boolean = false;
  private cassetteCrafted: boolean = false;
  private patchCrafted: boolean = false;
  private statusText!: Phaser.GameObjects.Text;

  constructor() {
    super({ key: 'BlueprintScene' });
  }

  create(): void {
    this.inventoryManager = InventoryManager.getInstance();
    this.runManager = RunManager.getInstance();
    this.uiManager = UIManager.getInstance();

    this.uiManager.updateSystemMessage('BLUEPRINT VIEW // SAFE ZONE');
    this.uiManager.updateHeat(this.runManager.getHeat());
    this.uiManager.updateStabilizer(this.runManager.getStabilizerCharges());

    // Ensure baseline scrap for EP1 machines
    if (this.inventoryManager.getScrapResources() < 6) {
      this.inventoryManager.addScrap(6 - this.inventoryManager.getScrapResources());
    }
    this.uiManager.getInventoryUI()?.update();

    const centerX = this.cameras.main.centerX;
    let posY = 80;

    this.add.text(centerX, posY, 'MAINTENANCE YARD // BUILD LINE', {
      fontFamily: 'VT323',
      fontSize: '26px',
      color: '#FFB000',
    }).setOrigin(0.5);

    posY += 50;
    this.statusText = this.add.text(centerX, posY, '', {
      fontFamily: 'VT323',
      fontSize: '18px',
      color: '#B026FF',
      align: 'center',
    }).setOrigin(0.5);

    posY += 40;
    this.buildButton(centerX, posY, 'Build Extractor (2 SCRAP)', () => this.buildMachine('extractor'));
    posY += 40;
    this.buildButton(centerX, posY, 'Build Refiner (2 SCRAP)', () => this.buildMachine('refiner'));
    posY += 40;
    this.buildButton(centerX, posY, 'Build Press (2 SCRAP)', () => this.buildMachine('press'));

    posY += 60;
    this.buildButton(centerX, posY, 'Craft Basic Attack Cassette', () => this.craftCassette());
    posY += 40;
    this.buildButton(centerX, posY, 'Craft Patch Kit', () => this.craftPatch());

    posY += 60;
    this.buildButton(centerX - 80, posY, 'Proceed: EXTRACT A', () => {
      this.runManager.setCurrentNodeById('N3A');
      this.scene.start('RunScene');
    });
    this.buildButton(centerX + 80, posY, 'Proceed: EXTRACT B', () => {
      this.runManager.setCurrentNodeById('N3B');
      this.scene.start('RunScene');
    });

    this.refreshStatus();
  }

  private buildButton(x: number, y: number, label: string, onClick: () => void): void {
    const button = this.add.text(x, y, label, {
      fontFamily: 'VT323',
      fontSize: '18px',
      color: '#FFFFFF',
      backgroundColor: '#3d2a00',
      padding: { left: 10, right: 10, top: 6, bottom: 6 },
    }).setOrigin(0.5);

    button.setInteractive({ useHandCursor: true });
    button.on('pointerdown', onClick);
  }

  private buildMachine(kind: 'extractor' | 'refiner' | 'press'): void {
    if (this.inventoryManager.getScrapResources() < 2) {
      this.uiManager.updateSystemMessage('INSUFFICIENT SCRAP');
      return;
    }

    this.inventoryManager.addScrap(-2);
    if (kind === 'extractor') this.extractorBuilt = true;
    if (kind === 'refiner') this.refinerBuilt = true;
    if (kind === 'press') this.pressBuilt = true;
    this.uiManager.getInventoryUI()?.update();
    this.refreshStatus();
  }

  private craftCassette(): void {
    if (!this.extractorBuilt || !this.refinerBuilt || !this.pressBuilt) {
      this.uiManager.updateSystemMessage('BUILD LINE FIRST');
      return;
    }
    if (this.cassetteCrafted) return;

    const cassette: ItemData = {
      id: `cassette-${Date.now()}`,
      name: 'BASIC ATTACK CASSETTE',
      type: ItemType.WEAPON,
      shape: [[1, 1]],
      rotation: 0,
      maxAmmo: 12,
      currentAmmo: 12,
      scrapValue: 8,
    };
    this.inventoryManager.addItemToTray(cassette);
    this.uiManager.getInventoryUI()?.update();
    this.cassetteCrafted = true;
    this.uiManager.updateSystemMessage('CASSETTE PRODUCED');
    this.refreshStatus();
  }

  private craftPatch(): void {
    if (!this.extractorBuilt || !this.refinerBuilt || !this.pressBuilt) {
      this.uiManager.updateSystemMessage('BUILD LINE FIRST');
      return;
    }
    if (this.patchCrafted) return;

    const patch: ItemData = {
      id: `patch-${Date.now()}`,
      name: 'PATCH KIT',
      type: ItemType.CONSUMABLE,
      shape: [[1]],
      rotation: 0,
      maxAmmo: null,
      currentAmmo: null,
      scrapValue: 4,
    };
    this.inventoryManager.addItemToTray(patch);
    this.uiManager.getInventoryUI()?.update();
    this.patchCrafted = true;
    this.uiManager.updateSystemMessage('PATCH KIT READY');
    this.refreshStatus();
  }

  private refreshStatus(): void {
    const lines: string[] = [];
    lines.push(`Extractor: ${this.extractorBuilt ? 'BUILT' : 'PENDING'} | Refiner: ${this.refinerBuilt ? 'BUILT' : 'PENDING'} | Press: ${this.pressBuilt ? 'BUILT' : 'PENDING'}`);
    lines.push(`Cassette: ${this.cassetteCrafted ? 'READY' : 'PENDING'} | Patch: ${this.patchCrafted ? 'READY' : 'PENDING'}`);

    if (this.cassetteCrafted && this.patchCrafted) {
      this.runManager.markProductionComplete();
      lines.push('Production objective complete. Choose extraction route.');
    } else {
      lines.push('Produce BASIC ATTACK CASSETTE + PATCH KIT to proceed.');
    }

    this.statusText.setText(lines.join('\n'));
  }
}
