import Phaser from 'phaser';
import { UIManager } from '../../ui/UIManager';
import { InventoryManager } from '../systems/InventoryManager';
import { RunManager } from '../systems/RunManager';
import { EnemyIntent, EnemyStats, NodeConfig } from '../types/Run';

interface Battler {
  id: string;
  name: string;
  hp: number;
  maxHp: number;
  move: number;
  range: number;
  damage: number;
  heal?: number;
  ap: number;
  gridX: number;
  gridY: number;
  sprite: Phaser.GameObjects.Rectangle;
  label: Phaser.GameObjects.Text;
  kind: 'ally' | 'enemy';
  enemyKind?: 'BLOCKER' | 'JAMMER';
}

interface CombatData {
  nodeId: string;
}

export class CombatScene extends Phaser.Scene {
  private uiManager!: UIManager;
  private inventoryManager!: InventoryManager;
  private runManager!: RunManager;
  private node!: NodeConfig;
  private grid: Phaser.GameObjects.Rectangle[][] = [];
  private readonly GRID_SIZE = 8;
  private readonly TILE_SIZE = 60;
  private allies: Battler[] = [];
  private enemies: Battler[] = [];
  private activeAllyIndex: number = 0;
  private currentMode: 'idle' | 'move' | 'attack' = 'idle';
  private highlightedTiles: Phaser.GameObjects.Rectangle[] = [];
  private intentMarkers: Phaser.GameObjects.Rectangle[] = [];

  private turnTimer: number = 30;
  private timerEvent?: Phaser.Time.TimerEvent;
  private isTimerActive: boolean = false;

  private extractionActive: boolean = false;
  private extractionStagesRequired: number = 0;
  private extractionZoneCenter?: { x: number; y: number };
  private extractionZoneRadius: number = 1;

  init(data: CombatData): void {
    this.node = RunManager.getInstance().getEpisode().nodes.find(n => n.id === data.nodeId)!;
  }

  create(): void {
    this.uiManager = UIManager.getInstance();
    this.inventoryManager = InventoryManager.getInstance();
    this.runManager = RunManager.getInstance();

    this.uiManager.updateSystemMessage(`COMBAT // ${this.node.name}`);
    this.uiManager.updateHeat(this.runManager.getHeat());
    this.uiManager.updateStabilizer(this.runManager.getStabilizerCharges());

    this.createGrid();
    this.createAllies();
    this.createEnemies();
    this.configureExtraction();
    this.bindControls();

    this.computeEnemyIntents();
    this.startPlayerTurn();
  }

  private createGrid(): void {
    const startX = this.cameras.main.centerX - (this.GRID_SIZE * this.TILE_SIZE) / 2;
    const startY = this.cameras.main.centerY - (this.GRID_SIZE * this.TILE_SIZE) / 2 + 20;

    for (let y = 0; y < this.GRID_SIZE; y++) {
      this.grid[y] = [];
      for (let x = 0; x < this.GRID_SIZE; x++) {
        const tile = this.add.rectangle(
          startX + x * this.TILE_SIZE + this.TILE_SIZE / 2,
          startY + y * this.TILE_SIZE + this.TILE_SIZE / 2,
          this.TILE_SIZE - 2,
          this.TILE_SIZE - 2,
          0x594100,
          0.2
        );
        tile.setStrokeStyle(1, 0xffb000, 0.5);
        tile.setInteractive({ useHandCursor: true });
        tile.setData('gridX', x);
        tile.setData('gridY', y);
        tile.on('pointerdown', () => this.handleTileClick(x, y));
        this.grid[y][x] = tile;
      }
    }
  }

  private createAllies(): void {
    const allyConfigs = [
      { id: 'io', name: 'I.O. DRONE', maxHp: 8, move: 5, range: 4, damage: 2, ap: 2, heal: 0 },
      { id: 'niko', name: 'NIKO ROVER', maxHp: 6, move: 4, range: 3, damage: 0, ap: 2, heal: 2 },
    ];

    const positions = [
      { x: 1, y: 1 },
      { x: 1, y: 2 },
    ];

    allyConfigs.forEach((cfg, index) => {
      const pos = positions[index];
      const tile = this.grid[pos.y][pos.x];
      const sprite = this.add.rectangle(tile.x, tile.y, 36, 36, 0x00ff00, 0.8);
      sprite.setStrokeStyle(2, 0xffffff);
      const label = this.add.text(tile.x, tile.y, cfg.name.split(' ')[0], {
        fontFamily: 'VT323',
        fontSize: '14px',
        color: '#000000',
      }).setOrigin(0.5);

      this.allies.push({
        id: cfg.id,
        name: cfg.name,
        hp: cfg.maxHp,
        maxHp: cfg.maxHp,
        move: cfg.move,
        range: cfg.range,
        damage: cfg.damage,
        heal: cfg.heal,
        ap: cfg.ap,
        gridX: pos.x,
        gridY: pos.y,
        sprite,
        label,
        kind: 'ally',
      });
    });
  }

  private createEnemies(): void {
    const encounter = this.node.encounter;
    if (!encounter) return;

    const positions = [
      { x: 6, y: 6 },
      { x: 6, y: 4 },
      { x: 5, y: 6 },
      { x: 6, y: 2 },
    ];

    encounter.enemies.forEach((enemy, index) => {
      const pos = positions[index] || { x: 6, y: 6 };
      this.spawnEnemy(enemy, pos.x, pos.y);
    });

    this.checkEnemyStates();
  }

  private spawnEnemy(enemy: EnemyStats, x: number, y: number): void {
    const tile = this.grid[y][x];
    const sprite = this.add.rectangle(tile.x, tile.y, 36, 36, 0xb026ff, 0.8);
    sprite.setStrokeStyle(2, 0xffffff);
    const label = this.add.text(tile.x, tile.y, enemy.name, {
      fontFamily: 'VT323',
      fontSize: '12px',
      color: '#ffffff',
      align: 'center',
    }).setOrigin(0.5);

    this.enemies.push({
      id: enemy.id,
      name: enemy.name,
      hp: enemy.maxHp,
      maxHp: enemy.maxHp,
      move: enemy.move,
      range: enemy.range,
      damage: enemy.damage,
      ap: enemy.ap,
      gridX: x,
      gridY: y,
      sprite,
      label,
      kind: 'enemy',
      enemyKind: enemy.kind,
    });
  }

  private configureExtraction(): void {
    const extraction = this.node.encounter?.extraction;
    if (!extraction) return;
    this.extractionActive = true;
    this.extractionZoneCenter = extraction.console;
    this.extractionZoneRadius = extraction.zoneRadius;
    const heatOffset = this.runManager.getHeat() >= 4 ? 1 : 0;
    this.extractionStagesRequired = extraction.baseStages + heatOffset;

    const tile = this.grid[extraction.console.y][extraction.console.x];
    this.add.circle(tile.x, tile.y, 10, 0xb026ff, 0.5);
  }

  private bindControls(): void {
    const btnMove = document.getElementById('btn-move');
    const btnAttack = document.getElementById('btn-attack');
    const btnInventory = document.getElementById('btn-inventory');
    const btnEndTurn = document.getElementById('btn-end-turn');

    if (btnMove) {
      btnMove.onclick = () => this.enterMoveMode();
    }
    if (btnAttack) {
      btnAttack.onclick = () => this.enterAttackMode();
    }
    if (btnInventory) {
      btnInventory.onclick = () => this.uiManager.toggleInventory();
    }
    if (btnEndTurn) {
      btnEndTurn.onclick = () => this.endPlayerTurn();
    }

    this.input.keyboard?.on('keydown-TAB', () => this.cycleActiveAlly());
    this.input.keyboard?.on('keydown-F', () => this.useStabilizer());
  }

  private cycleActiveAlly(): void {
    this.activeAllyIndex = (this.activeAllyIndex + 1) % this.allies.length;
    this.uiManager.updateSystemMessage(`ACTIVE: ${this.allies[this.activeAllyIndex].name}`);
  }

  private enterMoveMode(): void {
    this.currentMode = 'move';
    this.highlightedTiles = [];
    const ally = this.allies[this.activeAllyIndex];
    this.highlightRange(ally.gridX, ally.gridY, ally.move, 0x00ff00);
    this.uiManager.updateSystemMessage('SELECT MOVE TARGET');
  }

  private enterAttackMode(): void {
    this.currentMode = 'attack';
    this.highlightedTiles = [];
    const ally = this.allies[this.activeAllyIndex];
    this.highlightRange(ally.gridX, ally.gridY, ally.range, 0xff0000);
    this.uiManager.updateSystemMessage('SELECT ATTACK TARGET');
  }

  private highlightRange(x: number, y: number, range: number, color: number): void {
    this.clearHighlights();
    for (let row = 0; row < this.GRID_SIZE; row++) {
      for (let col = 0; col < this.GRID_SIZE; col++) {
        const distance = Math.abs(col - x) + Math.abs(row - y);
        if (distance > 0 && distance <= range) {
          const tile = this.grid[row][col];
          tile.setFillStyle(color, 0.3);
          tile.setStrokeStyle(2, color, 0.8);
          this.highlightedTiles.push(tile);
        }
      }
    }
  }

  private clearHighlights(): void {
    this.highlightedTiles.forEach(tile => {
      tile.setFillStyle(0x594100, 0.2);
      tile.setStrokeStyle(1, 0xffb000, 0.5);
    });
    this.highlightedTiles = [];
  }

  private handleTileClick(x: number, y: number): void {
    const ally = this.allies[this.activeAllyIndex];
    if (this.currentMode === 'move') {
      const distance = Math.abs(x - ally.gridX) + Math.abs(y - ally.gridY);
      if (distance <= ally.move && ally.ap > 0) {
        ally.ap -= 1;
        this.moveBattler(ally, x, y);
      }
      this.clearHighlights();
      this.currentMode = 'idle';
    } else if (this.currentMode === 'attack') {
      const distance = Math.abs(x - ally.gridX) + Math.abs(y - ally.gridY);
      if (distance <= ally.range && ally.ap > 0) {
        ally.ap -= 1;
        this.resolveAttack(ally, x, y);
      }
      this.clearHighlights();
      this.currentMode = 'idle';
    }
  }

  private moveBattler(battler: Battler, x: number, y: number): void {
    const tile = this.grid[y][x];
    this.tweens.add({
      targets: [battler.sprite, battler.label],
      x: tile.x,
      y: tile.y,
      duration: 180,
      onComplete: () => {
        battler.gridX = x;
        battler.gridY = y;
      },
    });
  }

  private resolveAttack(attacker: Battler, targetX: number, targetY: number): void {
    if (attacker.kind === 'ally' && attacker.heal && attacker.heal > 0) {
      const ally = this.allies.find(a => a.gridX === targetX && a.gridY === targetY);
      if (ally) {
        ally.hp = Math.min(ally.maxHp, ally.hp + attacker.heal);
        this.uiManager.updateSystemMessage(`PATCHED ${ally.name} +${attacker.heal}`);
      }
      return;
    }

    const enemy = this.enemies.find(e => e.gridX === targetX && e.gridY === targetY && e.hp > 0);
    if (!enemy) {
      this.uiManager.updateSystemMessage('NO TARGET');
      return;
    }

    enemy.hp -= attacker.damage;
    enemy.sprite.setFillStyle(0xff0000);
    this.time.delayedCall(150, () => enemy.sprite.setFillStyle(0xb026ff));
    if (enemy.hp <= 0) {
      enemy.sprite.destroy();
      enemy.label.destroy();
    }
    this.uiManager.updateSystemMessage(`${attacker.name} HIT ${enemy.name} FOR ${attacker.damage}`);
    this.checkEnemyStates();
  }

  private checkEnemyStates(): void {
    this.enemies = this.enemies.filter(e => e.hp > 0);
    const jammerAlive = this.enemies.some(e => e.enemyKind === 'JAMMER');
    this.inventoryManager.setJammerPenalty(jammerAlive);
    this.uiManager.getInventoryUI()?.update();

    if (this.enemies.length === 0 && !this.extractionActive) {
      this.finishNode();
    }
  }

  private useStabilizer(): void {
    if (this.runManager.consumeStabilizer()) {
      this.turnTimer += 10;
      this.uiManager.updateTimer(this.turnTimer);
      this.uiManager.updateStabilizer(this.runManager.getStabilizerCharges());
      this.uiManager.updateHeat(this.runManager.getHeat());
      this.uiManager.updateSystemMessage('STABILIZER ENGAGED +10s (HEAT +1)');
    }
  }

  private startPlayerTurn(): void {
    this.allies.forEach(ally => (ally.ap = 2));
    this.currentMode = 'idle';
    this.startTurnTimer();
    this.renderIntents();
    this.uiManager.updateSystemMessage('YOUR TURN');
  }

  private startTurnTimer(): void {
    this.turnTimer = 30;
    this.isTimerActive = true;
    this.uiManager.updateTimer(this.turnTimer);
    this.timerEvent?.remove();
    this.timerEvent = this.time.addEvent({ delay: 1000, loop: true, callback: this.updateTimer, callbackScope: this });
  }

  private updateTimer(): void {
    if (!this.isTimerActive) return;
    this.turnTimer -= 1;
    this.uiManager.updateTimer(this.turnTimer);
    if (this.turnTimer <= 5) {
      this.uiManager.setTimerWarning(true);
    }
    if (this.turnTimer <= 0) {
      this.handleTimeout();
    }
  }

  private handleTimeout(): void {
    this.isTimerActive = false;
    this.timerEvent?.remove();
    this.runManager.addHeat(1);
    this.uiManager.updateHeat(this.runManager.getHeat());
    this.uiManager.updateSystemMessage('SYSTEM HALTED // HEAT +1');
    this.time.delayedCall(400, () => this.endPlayerTurn());
  }

  private endPlayerTurn(): void {
    this.isTimerActive = false;
    this.timerEvent?.remove();
    this.uiManager.setTimerWarning(false);
    this.clearHighlights();
    this.resolveExtractionProgress();

    this.time.delayedCall(400, () => this.executeEnemyTurn());
  }

  private executeEnemyTurn(): void {
    this.uiManager.updateSystemMessage('ENEMY TURN');
    const intents = this.runManager.getPendingIntents();
    intents.forEach(intent => {
      const enemy = this.enemies.find(e => e.id === intent.enemyId);
      if (!enemy || enemy.hp <= 0) return;

      this.followIntent(enemy, intent);
    });

    this.computeEnemyIntents();
    this.startPlayerTurn();
  }

  private followIntent(enemy: Battler, intent: EnemyIntent): void {
    if (intent.type !== 'ATTACK_TILE') return;

    // Move closer if out of range
    const distance = Math.abs(enemy.gridX - intent.target.x) + Math.abs(enemy.gridY - intent.target.y);
    if (distance > enemy.range) {
      const steps = Math.min(enemy.move, distance - enemy.range);
      this.moveToward(enemy, intent.target.x, intent.target.y, steps);
    }

    const newDistance = Math.abs(enemy.gridX - intent.target.x) + Math.abs(enemy.gridY - intent.target.y);
    if (newDistance <= enemy.range) {
      const ally = this.allies.find(a => a.gridX === intent.target.x && a.gridY === intent.target.y && a.hp > 0);
      if (ally) {
        ally.hp -= enemy.damage;
        if (enemy.enemyKind === 'JAMMER') {
          this.runManager.addHeat(1);
          this.uiManager.updateHeat(this.runManager.getHeat());
        }
        if (ally.hp <= 0) {
          ally.sprite.destroy();
          ally.label.destroy();
        }
      }
    }
  }

  private moveToward(battler: Battler, targetX: number, targetY: number, steps: number): void {
    let x = battler.gridX;
    let y = battler.gridY;
    for (let i = 0; i < steps; i++) {
      if (x < targetX) x += 1;
      else if (x > targetX) x -= 1;
      else if (y < targetY) y += 1;
      else if (y > targetY) y -= 1;
    }
    this.moveBattler(battler, x, y);
  }

  private computeEnemyIntents(): void {
    const intents: EnemyIntent[] = [];
    this.enemies.filter(e => e.hp > 0).forEach(enemy => {
      const nearest = this.findNearestAlly(enemy.gridX, enemy.gridY);
      const target = nearest ? { x: nearest.gridX, y: nearest.gridY } : { x: enemy.gridX, y: enemy.gridY };
      intents.push({ enemyId: enemy.id, type: 'ATTACK_TILE', target });
    });
    this.runManager.setPendingIntents(intents);
  }

  private findNearestAlly(x: number, y: number): Battler | undefined {
    let nearest: Battler | undefined;
    let bestDistance = Number.MAX_SAFE_INTEGER;
    this.allies.filter(a => a.hp > 0).forEach(ally => {
      const distance = Math.abs(ally.gridX - x) + Math.abs(ally.gridY - y);
      if (distance < bestDistance) {
        bestDistance = distance;
        nearest = ally;
      }
    });
    return nearest;
  }

  private renderIntents(): void {
    this.intentMarkers.forEach(marker => marker.destroy());
    this.intentMarkers = [];
    const intents = this.runManager.getPendingIntents();
    intents.forEach(intent => {
      const tile = this.grid[intent.target.y][intent.target.x];
      const marker = this.add.rectangle(tile.x, tile.y, this.TILE_SIZE - 6, this.TILE_SIZE - 6, 0xff0000, 0.15);
      marker.setStrokeStyle(2, 0xff0000, 0.8);
      this.intentMarkers.push(marker);
    });
  }

  private resolveExtractionProgress(): void {
    if (!this.extractionActive || !this.extractionZoneCenter) return;
    const inZone = this.allies.some(ally => {
      const dist = Math.abs(ally.gridX - this.extractionZoneCenter!.x) + Math.abs(ally.gridY - this.extractionZoneCenter!.y);
      return dist <= this.extractionZoneRadius && ally.hp > 0;
    });
    if (inZone) {
      const stage = this.runManager.incrementExtractionStage(this.node.id);
      this.uiManager.updateSystemMessage(`UPLOAD STAGE ${stage}/${this.extractionStagesRequired}`);
      if (this.node.id === 'N1' && stage >= this.extractionStagesRequired) {
        this.runManager.markPowerRestored();
      }
      if (this.node.id === 'N3A' || this.node.id === 'N3B') {
        if (this.runManager.getHeat() >= 5 && this.node.id === 'N3B' && stage === 2) {
          this.spawnEnemy(
            {
              id: `blocker-reinforce-${Date.now()}`,
              name: 'BLOCKER',
              kind: 'BLOCKER',
              maxHp: 6,
              move: 3,
              range: 5,
              damage: 1,
              ap: 2,
            },
            4,
            4
          );
        }
      }
      if (stage >= this.extractionStagesRequired) {
        if (this.node.id === 'N3A' || this.node.id === 'N3B') {
          this.runManager.markExtractionComplete();
        }
        this.finishNode(true);
      }
    }
  }

  private finishNode(skipAdvance?: boolean): void {
    if (!skipAdvance && this.extractionActive) return;
    if (this.runManager.isRunComplete()) {
      this.runManager.setCurrentNodeById('RESULT');
      this.scene.start('ResultScene');
      return;
    }

    this.runManager.advanceNode();
    const nextNode = this.runManager.getCurrentNode();
    if (nextNode.type === 'combat') {
      this.scene.start('CombatScene', { nodeId: nextNode.id });
    } else if (nextNode.type === 'blueprint') {
      this.scene.start('BlueprintScene', { nodeId: nextNode.id });
    } else {
      this.scene.start('ResultScene');
    }
  }
}
