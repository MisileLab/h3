import Phaser from 'phaser';
import { InventoryManager } from '../systems/InventoryManager';
import { RunManager } from '../systems/RunManager';
import { StoryManager } from '../story/StoryManager';
import { UIManager } from '../../ui/UIManager';
import { ItemData, ItemType } from '../types/Item';

interface BlueprintData {
  nodeId?: string;
}

export class BlueprintScene extends Phaser.Scene {
  private inventoryManager!: InventoryManager;
  private runManager!: RunManager;
  private storyManager!: StoryManager;
  private uiManager!: UIManager;
  private extractorBuilt: boolean = false;
  private refinerBuilt: boolean = false;
  private pressBuilt: boolean = false;
  private cassetteCrafted: boolean = false;
  private patchCrafted: boolean = false;
  private coolantCrafted: boolean = false;
  private dewCollected: number = 0;
  private isEp2: boolean = false;
  private isEp3: boolean = false;
  private nodeId: string | null = null;
  private statusText!: Phaser.GameObjects.Text;

  init(data: BlueprintData): void {
    this.nodeId = data.nodeId ?? null;
  }

  constructor() {
    super({ key: 'BlueprintScene' });
  }

  create(): void {
    this.inventoryManager = InventoryManager.getInstance();
    this.runManager = RunManager.getInstance();
    this.storyManager = StoryManager.getInstance();
    this.storyManager.setEpisodeId(this.runManager.getEpisode().id);
    this.uiManager = UIManager.getInstance();

    this.isEp2 = this.runManager.getEpisode().id === 'ep2';
    this.isEp3 = this.runManager.getEpisode().id === 'ep3';

    if (!this.isEp2 && !this.isEp3 && this.nodeId === 'N2') {
      this.storyManager.trigger('TRG_N2_ENTER');
      this.storyManager.trigger('TRG_BLUEPRINT_INTRO');
    }

    this.uiManager.updateSystemMessage(
      this.isEp3
        ? this.nodeId === 'N0'
          ? 'SHELTER HUB // FIRST RULE'
          : 'RESCUE SAFE ZONE'
        : this.isEp2
          ? 'MAKER FLOOR // WORKSHOP LOCKDOWN'
          : 'BLUEPRINT VIEW // SAFE ZONE'
    );
    this.uiManager.updateHeat(this.runManager.getHeat());
    this.uiManager.updateStabilizer(this.runManager.getStabilizerCharges());

    if (!this.isEp2 && !this.isEp3) {
      // Ensure baseline scrap for EP1 machines
      if (this.inventoryManager.getScrapResources() < 6) {
        this.inventoryManager.addScrap(6 - this.inventoryManager.getScrapResources());
      }
    } else if (this.inventoryManager.getScrapResources() < 4) {
      this.inventoryManager.addScrap(4 - this.inventoryManager.getScrapResources());
    }
    this.uiManager.getInventoryUI()?.update();

    const centerX = this.cameras.main.centerX;
    let posY = 80;

    this.add.text(
      centerX,
      posY,
      this.isEp2
        ? 'COOLANT FAB // COLLECT + PRESS'
        : this.isEp3
          ? this.nodeId === 'N0'
            ? 'SHELTER RULE // PREP'
            : 'RESPITE // TRUST'
          : 'MAINTENANCE YARD // BUILD LINE',
      {
        fontFamily: 'VT323',
        fontSize: '26px',
        color: '#FFB000',
      }
    ).setOrigin(0.5);

    posY += 50;
    this.statusText = this.add.text(centerX, posY, '', {
      fontFamily: 'VT323',
      fontSize: '18px',
      color: '#B026FF',
      align: 'center',
    }).setOrigin(0.5);

    if (!this.isEp2 && !this.isEp3) {
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
    } else if (this.isEp2) {
      posY += 40;
      this.buildButton(centerX, posY, 'HARVEST RESONANT DEW (HEAT +1)', () => this.harvestDew());
      posY += 40;
      this.buildButton(centerX, posY, 'PRESS COOLANT (PLATE + DEW)', () => this.craftCoolant());
      posY += 60;
      this.buildButton(centerX, posY, 'Proceed: RELAY ROOM', () => {
        this.runManager.setCurrentNodeById('N3');
        this.scene.start('RunScene');
      });
    } else {
      posY += 30;
      if (this.nodeId === 'N0') {
        this.add.text(centerX, posY, 'FIRST RULE: DO NOT LEAVE STUDENTS ALONE', {
          fontFamily: 'VT323',
          fontSize: '18px',
          color: '#B026FF',
        }).setOrigin(0.5);
        posY += 50;
        this.buildButton(centerX, posY, 'Proceed to CLINIC RUN', () => {
          this.runManager.setCurrentNodeById('N1');
          this.scene.start('RunScene');
        });
      } else {
        this.runManager.markMedicJoined();
        this.add.text(centerX, posY, 'Rescue Medic joins. TRUST decision locked in.', {
          fontFamily: 'VT323',
          fontSize: '18px',
          color: '#B026FF',
        }).setOrigin(0.5);
        posY += 40;
        this.buildButton(centerX, posY, 'Craft Portable Barrier Kit (2 SCRAP)', () => this.craftBarrierKit());
        posY += 40;
        this.buildButton(centerX, posY, 'Craft Escort Strap (2 SCRAP)', () => this.craftEscortStrap());
        posY += 50;
        this.buildButton(centerX, posY, 'Proceed to ANCHOR APPROACH', () => {
          this.runManager.setCurrentNodeById('N3');
          this.scene.start('RunScene');
        });
      }
    }

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

    if (!this.isEp2 && !this.isEp3 && this.nodeId === 'N2') {
      if (kind === 'extractor') this.storyManager.trigger('TRG_PLACE_EXTRACTOR');
      if (kind === 'refiner') this.storyManager.trigger('TRG_PLACE_REFINER');
      if (kind === 'press') this.storyManager.trigger('TRG_PLACE_PRESS');

      if (this.extractorBuilt && this.refinerBuilt && this.pressBuilt) {
        this.storyManager.trigger('TRG_LINE_CONNECTED');
      }
    }

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

    if (!this.isEp2 && !this.isEp3 && this.nodeId === 'N2') {
      this.storyManager.trigger('TRG_PRODUCE_CASSETTE_FIRST');
    }

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

    if (!this.isEp2 && !this.isEp3 && this.nodeId === 'N2') {
      this.storyManager.trigger('TRG_PRODUCE_KIT_FIRST');
    }

    this.refreshStatus();
  }

  private harvestDew(): void {
    const dew: ItemData = {
      id: `dew-${Date.now()}`,
      name: 'RESONANT DEW',
      type: ItemType.CONSUMABLE,
      shape: [[1]],
      rotation: 0,
      maxAmmo: null,
      currentAmmo: null,
      scrapValue: 1,
    };
    this.inventoryManager.addItemToTray(dew);
    this.runManager.addHeat(1);
    this.uiManager.updateHeat(this.runManager.getHeat());
    this.dewCollected += 1;
    this.uiManager.getInventoryUI()?.update();
    this.uiManager.updateSystemMessage('RESONANT DEW CAPTURED (HEAT +1)');
    this.refreshStatus();
  }

  private craftCoolant(): void {
    if (this.inventoryManager.getScrapResources() < 1) {
      this.uiManager.updateSystemMessage('PLATE SHORTAGE: NEED 1 SCRAP');
      return;
    }

    const dewId = this.inventoryManager
      .getTray()
      .concat(this.inventoryManager.getBuffer())
      .find(item => item.name === 'RESONANT DEW')?.id;

    if (!dewId) {
      this.uiManager.updateSystemMessage('NEED RESONANT DEW');
      return;
    }

    this.inventoryManager.consumeItem(dewId);
    this.inventoryManager.addScrap(-1);

    const coolant: ItemData = {
      id: `coolant-${Date.now()}`,
      name: 'COOLANT CAPSULE',
      type: ItemType.CONSUMABLE,
      shape: [[1, 1]],
      rotation: 0,
      maxAmmo: null,
      currentAmmo: null,
      scrapValue: 2,
      description: 'Use in combat: +1 STAB or HEAT -1 (cap 2)',
    };
    this.inventoryManager.addItemToTray(coolant);
    this.coolantCrafted = true;
    this.runManager.markCoolantCrafted();
    this.uiManager.getInventoryUI()?.update();
    this.uiManager.updateSystemMessage('COOLANT READY');
    this.refreshStatus();
  }

  private craftBarrierKit(): void {
    if (this.inventoryManager.getScrapResources() < 2) {
      this.uiManager.updateSystemMessage('NEED 2 SCRAP');
      return;
    }
    this.inventoryManager.addScrap(-2);
    const kit: ItemData = {
      id: `barrier-kit-${Date.now()}`,
      name: 'PORTABLE BARRIER KIT',
      type: ItemType.CONSUMABLE,
      shape: [[1]],
      rotation: 0,
      maxAmmo: null,
      currentAmmo: null,
      scrapValue: 3,
      description: 'Deploy temporary cover in combat (1AP, 2 turns).',
    };
    this.inventoryManager.addItemToTray(kit);
    this.uiManager.getInventoryUI()?.update();
    this.uiManager.updateSystemMessage('BARRIER KIT READY');
  }

  private craftEscortStrap(): void {
    if (this.inventoryManager.getScrapResources() < 2) {
      this.uiManager.updateSystemMessage('NEED 2 SCRAP');
      return;
    }
    this.inventoryManager.addScrap(-2);
    const strap: ItemData = {
      id: `escort-strap-${Date.now()}`,
      name: 'ESCORT STRAP',
      type: ItemType.CONSUMABLE,
      shape: [[1]],
      rotation: 0,
      maxAmmo: null,
      currentAmmo: null,
      scrapValue: 2,
      description: 'Use on rescued ally for push resistance.',
    };
    this.inventoryManager.addItemToTray(strap);
    this.uiManager.getInventoryUI()?.update();
    this.uiManager.updateSystemMessage('ESCORT STRAP READY');
  }

  private refreshStatus(): void {
    const lines: string[] = [];
    if (!this.isEp2 && !this.isEp3) {
      lines.push(`Extractor: ${this.extractorBuilt ? 'BUILT' : 'PENDING'} | Refiner: ${this.refinerBuilt ? 'BUILT' : 'PENDING'} | Press: ${this.pressBuilt ? 'BUILT' : 'PENDING'}`);
      lines.push(`Cassette: ${this.cassetteCrafted ? 'READY' : 'PENDING'} | Patch: ${this.patchCrafted ? 'READY' : 'PENDING'}`);

      if (this.cassetteCrafted && this.patchCrafted) {
        this.runManager.markProductionComplete();
        if (this.nodeId === 'N2') {
          this.storyManager.trigger('TRG_N2_OBJECTIVE_COMPLETE');
          this.storyManager.trigger('TRG_EXTRACT_CHOICE');
        }
        lines.push('Production objective complete. Choose extraction route.');
      } else {
        lines.push('Produce BASIC ATTACK CASSETTE + PATCH KIT to proceed.');
      }
    } else if (this.isEp2) {
      lines.push(`Resonant Dew: ${this.dewCollected}`);
      lines.push(`Coolant: ${this.coolantCrafted ? 'READY' : 'PENDING'}`);
      lines.push('Goal: Craft at least one COOLANT CAPSULE.');
      if (this.coolantCrafted) {
        this.runManager.markProductionComplete();
        lines.push('Proceed to Relay Room.');
      }
    } else {
      lines.push('Escort prep: keep Medic safe, disable anchor, extract.');
      lines.push(`Tone: ${this.runManager.getToneFlag() ?? 'DEFAULT TRUST'}`);
      lines.push('Optional crafts: Barrier Kit, Escort Strap.');
    }

    this.statusText.setText(lines.join('\n'));
  }
}
