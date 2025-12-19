import Phaser from 'phaser';
import { UIManager } from '../../ui/UIManager';
import { StoryManager } from '../story/StoryManager';
import { InventoryManager } from '../systems/InventoryManager';
import { RunManager } from '../systems/RunManager';
import { ItemData, ItemType } from '../types/Item';
import { EnemyIntent, EnemyStats, IntentTarget, NodeConfig } from '../types/Run';

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
  enemyKind?: 'BLOCKER' | 'JAMMER' | 'HUNTER' | 'WARDEN';
}

interface CombatData {
  nodeId: string;
}

export class CombatScene extends Phaser.Scene {
  constructor() {
    super({ key: 'CombatScene' });
  }

  private uiManager!: UIManager;
  private storyManager!: StoryManager;
  private inventoryManager!: InventoryManager;
  private runManager!: RunManager;
  private jammerWasAlive: boolean = false;
  private node!: NodeConfig;
  private grid: Phaser.GameObjects.Rectangle[][] = [];
  private readonly GRID_SIZE = 8;
  private readonly TILE_SIZE = 60;
  private allies: Battler[] = [];
  private enemies: Battler[] = [];
  private activeAllyIndex: number = 0;
  private currentMode: 'idle' | 'move' | 'attack' | 'shield' = 'idle';
  private highlightedTiles: Phaser.GameObjects.Rectangle[] = [];
  private intentMarkers: Phaser.GameObjects.Rectangle[] = [];

  private turnTimer: number = 30;
  private timerEvent?: Phaser.Time.TimerEvent;
  private isTimerActive: boolean = false;

  private extractionActive: boolean = false;
  private extractionStagesRequired: number = 0;
  private extractionZoneCenter?: { x: number; y: number };
  private extractionZoneRadius: number = 1;
  private barriers: Map<string, Phaser.GameObjects.Rectangle> = new Map();
  private rescueUnjamUsed: boolean = false;
  private medicHealUsed: boolean = false;
  private antiphaseUses: number = 0;
  private shieldedUnits: Set<string> = new Set();
  private resonanceNodes: IntentTarget[] = [];
  private surgeTelegraph?: IntentTarget;
  private surgeMarkers: Phaser.GameObjects.Rectangle[] = [];
  private surgeIndex: number = 0;
  private anchorProgress: number = 0;
  private anchorRequired: number = 0;
  private anchorConsole?: IntentTarget;

  init(data: CombatData): void {
    this.node = RunManager.getInstance().getEpisode().nodes.find(n => n.id === data.nodeId)!;
  }

  create(): void {
    this.uiManager = UIManager.getInstance();
    this.inventoryManager = InventoryManager.getInstance();
    this.runManager = RunManager.getInstance();
    this.storyManager = StoryManager.getInstance();
    this.storyManager.setEpisodeId(this.runManager.getEpisode().id);
    this.triggerNodeEnterStory();

    this.uiManager.updateSystemMessage(`COMBAT // ${this.node.name}`);
    this.uiManager.updateHeat(this.runManager.getHeat());
    this.uiManager.updateStabilizer(this.runManager.getStabilizerCharges());

    if (this.node.extractionHeatSpike) {
      this.runManager.addHeat(this.node.extractionHeatSpike);
      this.uiManager.updateHeat(this.runManager.getHeat());
      this.uiManager.updateSystemMessage(`ENTRY HEAT +${this.node.extractionHeatSpike}`);
    }

    this.createGrid();
    this.createAllies();
    this.createEnemies();
    this.configureExtraction();
    this.configureResonance();
    this.bindControls();

    this.computeEnemyIntents();
    this.startPlayerTurn();
  }

  private isEp1(): boolean {
    return this.runManager.getEpisode().id === 'ep1';
  }

  private triggerNodeEnterStory(): void {
    if (!this.isEp1()) return;

    if (this.node.id === 'N0') {
      this.storyManager.trigger('TRG_N0_ENTER');
      return;
    }

    if (this.node.id === 'N1') {
      this.storyManager.trigger('TRG_N1_ENTER');
      this.storyManager.trigger('TRG_STAB_GIVE');
      return;
    }

    if (this.node.id === 'N3A') {
      this.storyManager.trigger('TRG_N3A_ENTER');
      return;
    }

    if (this.node.id === 'N3B') {
      this.storyManager.trigger('TRG_N3B_ENTER');
    }
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

    if (this.runManager.hasRescueJoined()) {
      allyConfigs.push({ id: 'rescue', name: 'RESCUE STUDENT', maxHp: 5, move: 4, range: 2, damage: 0, ap: 2, heal: 0 });
    }

    if (this.runManager.hasMedicJoined()) {
      allyConfigs.push({ id: 'medic', name: 'RESCUE MEDIC', maxHp: 5, move: 4, range: 3, damage: 0, ap: 2, heal: 2 });
    }

    const positions = [
      { x: 1, y: 1 },
      { x: 1, y: 2 },
      { x: 1, y: 3 },
      { x: 1, y: 4 },
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

    if (this.isEp1() && this.node.id === 'N1') {
      this.storyManager.trigger('TRG_BREAKER_SEEN');
    }
  }

  private configureResonance(): void {
    if (!this.node.encounter) return;
    this.resonanceNodes = this.node.encounter.resonanceNodes ?? [];
    this.anchorConsole = this.node.encounter.anchorConsole;
    this.anchorRequired = this.node.encounter.anchorStagesRequired ?? 0;
    if (this.anchorConsole) {
      const tile = this.grid[this.anchorConsole.y][this.anchorConsole.x];
      this.add.rectangle(tile.x, tile.y, this.TILE_SIZE - 8, this.TILE_SIZE - 8, 0x0088cc, 0.2).setStrokeStyle(2, 0x00ffff, 0.8);
    }
  }

  private bindControls(): void {
    const btnMove = document.getElementById('btn-move');
    const btnAttack = document.getElementById('btn-attack');
    const btnSwitch = document.getElementById('btn-switch');
    const btnInventory = document.getElementById('btn-inventory');
    const btnEndTurn = document.getElementById('btn-end-turn');

    if (btnMove) {
      btnMove.onclick = () => this.enterMoveMode();
    }
    if (btnAttack) {
      btnAttack.onclick = () => this.enterAttackMode();
    }
    if (btnSwitch) {
      btnSwitch.onclick = () => this.cycleActiveAlly();
    }
    if (btnInventory) {
      btnInventory.onclick = () => this.toggleInventory();
    }
    if (btnEndTurn) {
      btnEndTurn.onclick = () => this.endPlayerTurn();
    }

    this.input.keyboard?.on('keydown-TAB', (event: KeyboardEvent) => {
      event.preventDefault();
      this.toggleInventory();
    });
    this.input.keyboard?.on('keydown-C', () => this.cycleActiveAlly());
    this.input.keyboard?.on('keydown-F', () => this.useStabilizer());
    this.input.keyboard?.on('keydown-G', () => this.enterShieldMode());
  }

  private toggleInventory(): void {
    this.uiManager.toggleInventory();

    if (this.isEp1() && this.node.id === 'N0') {
      this.storyManager.trigger('TRG_TRAY_EXPLAIN');
      this.storyManager.trigger('TRG_TETRIS_EXPLAIN');
      this.storyManager.trigger('TRG_HEAT_INTRO');
    }
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

    if (this.isEp1() && this.node.id === 'N0') {
      this.storyManager.trigger('TRG_TUT_NORNG_FIRST');
    }
  }

  private enterShieldMode(): void {
    const ally = this.allies[this.activeAllyIndex];
    if (ally.id !== 'medic') {
      this.uiManager.updateSystemMessage('ANTIPHASE SHIELD: MEDIC ONLY');
      return;
    }
    this.currentMode = 'shield';
    this.highlightRange(ally.gridX, ally.gridY, ally.range, 0x00b0ff);
    this.uiManager.updateSystemMessage('SELECT SHIELD TARGET');
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
      if (this.anchorConsole && x === this.anchorConsole.x && y === this.anchorConsole.y) {
        const distance = Math.abs(x - ally.gridX) + Math.abs(y - ally.gridY);
        if (distance <= 1 && ally.ap > 0) {
          ally.ap -= 1;
          this.anchorProgress += 1;
          this.uiManager.updateSystemMessage(`ANCHOR DISABLE ${this.anchorProgress}/${this.anchorRequired || 3}`);
          if (this.anchorProgress >= (this.anchorRequired || 3)) {
            this.runManager.markAnchorDisabled();
            this.uiManager.updateSystemMessage('ANCHOR DISABLED');
            this.finishNode(true);
            return;
          }
        }
        this.clearHighlights();
        this.currentMode = 'idle';
        return;
      }
      const distance = Math.abs(x - ally.gridX) + Math.abs(y - ally.gridY);
      if (distance <= ally.range && ally.ap > 0) {
        ally.ap -= 1;
        if (ally.id === 'rescue') {
          this.useUnjam(x, y);
        } else if (ally.id === 'medic') {
          this.applyTriage(ally, x, y);
        } else {
          this.resolveAttack(ally, x, y);
        }
      }
      this.clearHighlights();
      this.currentMode = 'idle';
    } else if (this.currentMode === 'shield') {
      const distance = Math.abs(x - ally.gridX) + Math.abs(y - ally.gridY);
      if (ally.id === 'medic' && distance <= ally.range && ally.ap > 0) {
        this.applyAntiphaseShield(ally, x, y);
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

  private applyTriage(attacker: Battler, targetX: number, targetY: number): void {
    const ally = this.allies.find(a => a.gridX === targetX && a.gridY === targetY && a.hp > 0);
    if (!ally) {
      this.uiManager.updateSystemMessage('TRIAGE: NEED ALLY TARGET');
      return;
    }
    if (this.medicHealUsed) {
      this.uiManager.updateSystemMessage('TRIAGE ALREADY USED');
      return;
    }
    ally.hp = Math.min(ally.maxHp, ally.hp + (attacker.heal ?? 2));
    this.medicHealUsed = true;
    this.uiManager.updateSystemMessage(`TRIAGE PATCH +${attacker.heal ?? 2}`);
  }

  private applyAntiphaseShield(attacker: Battler, targetX: number, targetY: number): void {
    const ally = this.allies.find(a => a.gridX === targetX && a.gridY === targetY && a.hp > 0);
    if (!ally) {
      this.uiManager.updateSystemMessage('ANTIPHASE: NEED ALLY');
      return;
    }
    if (this.antiphaseUses >= 2) {
      this.uiManager.updateSystemMessage('ANTIPHASE LIMIT REACHED');
      return;
    }
    attacker.ap -= 1;
    this.antiphaseUses += 1;
    this.shieldedUnits.add(ally.id);
    this.uiManager.updateSystemMessage(`ANTIPHASE ON ${ally.name}`);
  }

  private useUnjam(targetX: number, targetY: number): void {
    if (this.rescueUnjamUsed) {
      this.uiManager.updateSystemMessage('UNJAM ALREADY USED');
      return;
    }
    const key = `${targetX},${targetY}`;
    const barrier = this.barriers.get(key);
    if (barrier) {
      barrier.destroy();
      this.barriers.delete(key);
      this.rescueUnjamUsed = true;
      this.uiManager.updateSystemMessage('BARRIER REMOVED');
      return;
    }
    this.uiManager.updateSystemMessage('NO BARRIER TO REMOVE');
  }

  private checkEnemyStates(): void {
    this.enemies = this.enemies.filter(e => e.hp > 0);
    const jammerAlive = this.enemies.some(e => e.enemyKind === 'JAMMER');

    if (this.isEp1() && this.node.id === 'N1' && jammerAlive && !this.jammerWasAlive) {
      this.storyManager.trigger('TRG_JAMMER_SPAWN_FIRST');
    }
    this.jammerWasAlive = jammerAlive;

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

      if (this.isEp1() && this.node.id === 'N1') {
        this.storyManager.trigger('TRG_STAB_USE_FIRST');
      }
    }
  }

  private startPlayerTurn(): void {
    this.allies.forEach(ally => (ally.ap = 2));
    this.rescueUnjamUsed = false;
    this.medicHealUsed = false;
    this.currentMode = 'idle';
    this.resolveResonanceSurge();
    this.startTurnTimer();
    this.renderIntents();

    if (this.isEp1() && this.node.id === 'N0') {
      this.storyManager.trigger('TRG_TUT_INTENT_FIRST');
    }

    this.uiManager.updateSystemMessage('YOUR TURN');
  }

  private startTurnTimer(): void {
    this.turnTimer = 30;
    this.isTimerActive = true;
    this.uiManager.updateTimer(this.turnTimer);

    if (this.isEp1() && this.node.id === 'N0') {
      this.storyManager.trigger('TRG_TUT_TIMER_START');
    }

    this.timerEvent?.remove();
    this.timerEvent = this.time.addEvent({ delay: 1000, loop: true, callback: this.updateTimer, callbackScope: this });
  }

  private updateTimer(): void {
    if (!this.isTimerActive) return;
    this.turnTimer -= 1;
    this.uiManager.updateTimer(this.turnTimer);

    if (this.turnTimer === 10 && this.isEp1() && this.node.id === 'N0') {
      this.storyManager.trigger('TRG_TIMER_10S');
    }

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

    if (this.isEp1() && this.node.id === 'N0') {
      this.storyManager.trigger('TRG_HALTED_FIRST');
    }

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

    this.telegraphSurge();
    this.computeEnemyIntents();
    this.startPlayerTurn();
  }

  private followIntent(enemy: Battler, intent: EnemyIntent): void {
    if (intent.type === 'PLACE_BARRIER') {
      const key = `${intent.target.x},${intent.target.y}`;
      const occupied = this.allies.some(a => a.gridX === intent.target.x && a.gridY === intent.target.y) ||
        this.enemies.some(e => e.gridX === intent.target.x && e.gridY === intent.target.y);
      if (!occupied && !this.barriers.has(key)) {
        const tile = this.grid[intent.target.y][intent.target.x];
        const rect = this.add.rectangle(tile.x, tile.y, this.TILE_SIZE - 12, this.TILE_SIZE - 12, 0x594100, 0.9);
        rect.setStrokeStyle(2, 0xffb000, 1);
        this.barriers.set(key, rect);
        this.uiManager.updateSystemMessage('BLOCKER DEPLOYED TAPE BARRIER');
      }
      return;
    }

    if (intent.type === 'POUNCE') {
      this.resolvePounce(enemy, intent.target.x, intent.target.y);
      return;
    }

    if (intent.type === 'ANCHOR_PULSE') {
      this.resolveAnchorPulse(enemy, intent.target.x, intent.target.y);
      return;
    }

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

          if (this.isEp1() && this.node.id === 'N1') {
            this.storyManager.trigger('TRG_JAMMER_EMP_HIT_FIRST');
          }
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

  private resolvePounce(enemy: Battler, targetX: number, targetY: number): void {
    let x = enemy.gridX;
    let y = enemy.gridY;
    for (let i = 0; i < enemy.move; i++) {
      if (x < targetX) x += 1;
      else if (x > targetX) x -= 1;
      else if (y < targetY) y += 1;
      else if (y > targetY) y -= 1;
    }
    enemy.gridX = x;
    enemy.gridY = y;
    const tile = this.grid[y][x];
    enemy.sprite.setPosition(tile.x, tile.y);
    enemy.label.setPosition(tile.x, tile.y);

    const ally = this.allies.find(a => a.gridX === enemy.gridX && a.gridY === enemy.gridY && a.hp > 0);
    if (ally) {
      ally.hp -= enemy.damage;
      const dx = ally.gridX - enemy.gridX;
      const dy = ally.gridY - enemy.gridY;
      const pushX = ally.gridX + Math.sign(dx || 1);
      const pushY = ally.gridY + Math.sign(dy || 0);
      if (pushX >= 0 && pushX < this.GRID_SIZE && pushY >= 0 && pushY < this.GRID_SIZE) {
        ally.gridX = pushX;
        ally.gridY = pushY;
        const tile = this.grid[pushY][pushX];
        ally.sprite.setPosition(tile.x, tile.y);
        ally.label.setPosition(tile.x, tile.y);
      }
      if (ally.hp <= 0) {
        ally.sprite.destroy();
        ally.label.destroy();
      }
      this.uiManager.updateSystemMessage('HUNTER POUNCE IMPACT');
    }
  }

  private resolveAnchorPulse(enemy: Battler, targetX: number, targetY: number): void {
    const distance = Math.abs(enemy.gridX - targetX) + Math.abs(enemy.gridY - targetY);
    if (distance > enemy.range) {
      const steps = Math.min(enemy.move, distance - enemy.range);
      this.moveToward(enemy, targetX, targetY, steps);
    }
    const ally = this.allies.find(a => a.gridX === targetX && a.gridY === targetY && a.hp > 0);
    if (ally) {
      ally.hp -= enemy.damage;
      const pushY = Math.min(this.GRID_SIZE - 1, ally.gridY + 1);
      ally.gridY = pushY;
      const tile = this.grid[ally.gridY][ally.gridX];
      ally.sprite.setPosition(tile.x, tile.y);
      ally.label.setPosition(tile.x, tile.y);
      if (ally.hp <= 0) {
        ally.sprite.destroy();
        ally.label.destroy();
      }
    }
    this.uiManager.updateSystemMessage('ANCHOR PULSE DETONATED');
  }

  private computeEnemyIntents(): void {
    const intents: EnemyIntent[] = [];
    this.enemies.filter(e => e.hp > 0).forEach(enemy => {
      const nearest = this.findNearestAlly(enemy.gridX, enemy.gridY);
      const target = nearest ? { x: nearest.gridX, y: nearest.gridY } : { x: enemy.gridX, y: enemy.gridY };
      if (enemy.enemyKind === 'HUNTER') {
        intents.push({ enemyId: enemy.id, type: 'POUNCE', target });
      } else if (enemy.enemyKind === 'WARDEN') {
        const focus = this.anchorConsole ?? target;
        intents.push({ enemyId: enemy.id, type: 'ANCHOR_PULSE', target: focus });
      } else if (enemy.enemyKind === 'BLOCKER' && Math.abs(enemy.gridX - target.x) + Math.abs(enemy.gridY - target.y) > enemy.range) {
        intents.push({ enemyId: enemy.id, type: 'PLACE_BARRIER', target });
      } else {
        intents.push({ enemyId: enemy.id, type: 'ATTACK_TILE', target });
      }
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

  private renderSurgeTelegraph(): void {
    this.surgeMarkers.forEach(marker => marker.destroy());
    this.surgeMarkers = [];
    if (!this.surgeTelegraph) return;
    const tile = this.grid[this.surgeTelegraph.y][this.surgeTelegraph.x];
    const marker = this.add.rectangle(tile.x, tile.y, this.TILE_SIZE - 10, this.TILE_SIZE - 10, 0x00ffff, 0.15);
    marker.setStrokeStyle(2, 0x00ffff, 0.8);
    this.surgeMarkers.push(marker);
  }

  private handleEp1ExtractionStory(stage: number): void {
    if (!this.isEp1()) return;

    if (this.node.id === 'N1') {
      if (stage === 1) {
        this.storyManager.trigger('TRG_BREAKER_UPLOAD_1');
      }
      if (stage >= this.extractionStagesRequired) {
        this.storyManager.trigger('TRG_BREAKER_UPLOAD_2');
      }
      return;
    }

    if (this.node.id === 'N3A') {
      if (stage === 1) {
        this.storyManager.trigger('TRG_UPLOAD_START_A');
        this.storyManager.trigger('TRG_UPLOAD_STEP_A_1');
      }
      if (this.runManager.getHeat() >= 4) {
        this.storyManager.trigger('TRG_UPLOAD_HEAT_WARN');
      }
      if (stage >= this.extractionStagesRequired) {
        this.storyManager.trigger('TRG_UPLOAD_COMPLETE_A');
      }
      return;
    }

    if (this.node.id === 'N3B') {
      if (stage === 1) {
        this.storyManager.trigger('TRG_UPLOAD_START_B');
        this.storyManager.trigger('TRG_UPLOAD_STEP_B_1');
      }
      if (stage === 2) {
        this.storyManager.trigger('TRG_UPLOAD_STEP_B_2');
        this.storyManager.trigger('TRG_RESCUE_THREAD_UNLOCK');
      }
      if (this.runManager.getHeat() >= 4) {
        this.storyManager.trigger('TRG_UPLOAD_HEAT_WARN_B');
      }
      if (stage >= this.extractionStagesRequired) {
        this.storyManager.trigger('TRG_UPLOAD_COMPLETE_B');
      }
    }
  }

  private resolveExtractionProgress(): void {
    if (!this.extractionActive || !this.extractionZoneCenter) return;
    const inZone = this.allies.some(ally => {
      const dist = Math.abs(ally.gridX - this.extractionZoneCenter!.x) + Math.abs(ally.gridY - this.extractionZoneCenter!.y);
      return dist <= this.extractionZoneRadius && ally.hp > 0;
    });
    const medicRequired = this.runManager.hasMedicJoined();
    const medicInZone = this.allies.some(
      ally =>
        ally.id === 'medic' &&
        ally.hp > 0 &&
        Math.abs(ally.gridX - this.extractionZoneCenter!.x) + Math.abs(ally.gridY - this.extractionZoneCenter!.y) <= this.extractionZoneRadius
    );
    if (medicRequired && !medicInZone) {
      return;
    }
    if (inZone) {
      const stage = this.runManager.incrementExtractionStage(this.node.id);
      this.uiManager.updateSystemMessage(`UPLOAD STAGE ${stage}/${this.extractionStagesRequired}`);
      this.handleEp1ExtractionStory(stage);
      if (this.node.id === 'N1' && stage >= this.extractionStagesRequired) {
        this.runManager.markPowerRestored();
      }
      if (this.node.id === 'N3A' || this.node.id === 'N3B' || this.node.id === 'N4A' || this.node.id === 'N4B') {
        if (this.runManager.getHeat() >= 5 && (this.node.id === 'N3B' || this.node.id === 'N4B') && stage === 2) {
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
        if (
          this.node.id === 'N3A' ||
          this.node.id === 'N3B' ||
          this.node.id === 'N4A' ||
          this.node.id === 'N4B' ||
          this.node.id === 'N5A' ||
          this.node.id === 'N5B'
        ) {
          this.runManager.markExtractionComplete(this.node.id);
        }
        this.finishNode(true);
      }
    }
  }

  private telegraphSurge(): void {
    if (!this.resonanceNodes.length || this.runManager.getHeat() < 7) {
      this.surgeTelegraph = undefined;
      this.renderSurgeTelegraph();
      return;
    }
    const target = this.resonanceNodes[this.surgeIndex % this.resonanceNodes.length];
    this.surgeIndex += 1;
    this.surgeTelegraph = target;
    this.uiManager.updateSystemMessage('RESONANCE SURGE INCOMING');
    this.renderSurgeTelegraph();
  }

  private resolveResonanceSurge(): void {
    if (!this.surgeTelegraph) return;
    const tile = this.grid[this.surgeTelegraph.y][this.surgeTelegraph.x];
    this.uiManager.updateSystemMessage('RESONANCE SURGE');
    const impacted = this.allies.filter(a => a.gridX === this.surgeTelegraph!.x && a.gridY === this.surgeTelegraph!.y && a.hp > 0);
    impacted.forEach(ally => {
      if (this.shieldedUnits.has(ally.id)) {
        this.shieldedUnits.delete(ally.id);
        this.uiManager.updateSystemMessage(`ANTIPHASE BLOCKED SURGE ON ${ally.name}`);
        return;
      }
      ally.hp -= 1;
      if (ally.hp <= 0) {
        ally.sprite.destroy();
        ally.label.destroy();
      } else {
        const pushY = Math.min(this.GRID_SIZE - 1, ally.gridY + 1);
        ally.gridY = pushY;
        const pushTile = this.grid[pushY][ally.gridX];
        ally.sprite.setPosition(pushTile.x, pushTile.y);
        ally.label.setPosition(pushTile.x, pushTile.y);
      }
    });
    const surgeMark = this.add.text(tile.x, tile.y, 'SURGE', {
      fontFamily: 'VT323',
      fontSize: '12px',
      color: '#00ffff',
    }).setOrigin(0.5);
    this.time.delayedCall(400, () => surgeMark.destroy());
    this.surgeTelegraph = undefined;
    this.renderSurgeTelegraph();
  }

  private finishNode(skipAdvance?: boolean): void {
    if (!skipAdvance && this.extractionActive) return;

    if (this.isEp1() && this.node.id === 'N0') {
      this.storyManager.trigger('TRG_N0_CLEAR');
    }
    if (this.node.id === 'N1' && this.runManager.getEpisode().id === 'ep2') {
      this.runManager.markRescueJoined();
      this.uiManager.updateSystemMessage('RESCUE STUDENT JOINED');
    }
    if (this.node.id === 'N1' && this.runManager.getEpisode().id === 'ep3') {
      this.runManager.markMedicJoined();
      this.uiManager.updateSystemMessage('RESCUE MEDIC FREED');
    }
    if (this.node.id === 'N3' && this.runManager.getEpisode().id === 'ep2') {
      this.runManager.markRelaySecured();
      const relayCore: ItemData = {
        id: `relay-core-${Date.now()}`,
        name: 'RELAY CORE',
        type: ItemType.CONSUMABLE,
        shape: [[1, 1]],
        rotation: 0,
        maxAmmo: null,
        currentAmmo: null,
        scrapValue: 5,
        description: 'Recovered alien relay core.',
      };
      this.inventoryManager.addItemToTray(relayCore);
      this.uiManager.getInventoryUI()?.update();
    }
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
