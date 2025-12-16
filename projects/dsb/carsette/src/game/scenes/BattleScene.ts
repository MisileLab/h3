import Phaser from 'phaser';
import { UIManager } from '../../ui/UIManager';
import { InventoryManager } from '../systems/InventoryManager';
import { ItemData, ItemType } from '../types/Item';

export class BattleScene extends Phaser.Scene {
  private uiManager!: UIManager;
  private inventoryManager!: InventoryManager;
  private turnTimer: number = 30;
  private timerEvent?: Phaser.Time.TimerEvent;
  private isTimerActive: boolean = false;
  private grid: Phaser.GameObjects.Rectangle[][] = [];
  private readonly GRID_SIZE = 8;
  private readonly TILE_SIZE = 50;
  
  // Player tracking
  private playerSprite!: Phaser.GameObjects.Arc;
  private playerText!: Phaser.GameObjects.Text;
  private playerGridX: number = 2;
  private playerGridY: number = 2;
  private readonly MOVE_RANGE: number = 3;
  private readonly ATTACK_RANGE: number = 4;
  
  // Enemy tracking
  private enemies: Array<{
    sprite: Phaser.GameObjects.Arc;
    text: Phaser.GameObjects.Text;
    gridX: number;
    gridY: number;
    hp: number;
    maxHp: number;
  }> = [];
  
  // Item tracking
  private fieldItems: Array<{
    sprite: Phaser.GameObjects.Container;
    gridX: number;
    gridY: number;
    itemData: ItemData;
  }> = [];
  
  // Game state
  private currentMode: 'idle' | 'moving' | 'attacking' = 'idle';
  private highlightedTiles: Phaser.GameObjects.Rectangle[] = [];

  constructor() {
    super({ key: 'BattleScene' });
  }

  create(): void {
    this.uiManager = UIManager.getInstance();
    this.inventoryManager = InventoryManager.getInstance();
    this.uiManager.updateSystemMessage('BATTLE MODE ACTIVE');

    // Draw battle grid
    this.createGrid();

    // Create player placeholder
    this.createPlayer(2, 2);

    // Create enemy placeholder
    this.createEnemy(6, 6);
    
    // Create test items on map
    this.createTestItems();

    // Set up control buttons
    this.setupControlButtons();

    // Start the turn timer
    this.startTurnTimer();

    // Title text
    const titleText = this.add.text(
      this.cameras.main.centerX,
      20,
      'BATTLE // TURN-BASED MODE',
      {
        fontFamily: 'VT323',
        fontSize: '24px',
        color: '#FFB000',
      }
    );
    titleText.setOrigin(0.5);
  }

  private createGrid(): void {
    const centerX = this.cameras.main.centerX;
    const centerY = this.cameras.main.centerY;
    const gridWidth = this.GRID_SIZE * this.TILE_SIZE;
    const gridHeight = this.GRID_SIZE * this.TILE_SIZE;
    const startX = centerX - gridWidth / 2;
    const startY = centerY - gridHeight / 2 + 20;

    for (let row = 0; row < this.GRID_SIZE; row++) {
      this.grid[row] = [];
      for (let col = 0; col < this.GRID_SIZE; col++) {
        const x = startX + col * this.TILE_SIZE + this.TILE_SIZE / 2;
        const y = startY + row * this.TILE_SIZE + this.TILE_SIZE / 2;

        const tile = this.add.rectangle(
          x,
          y,
          this.TILE_SIZE - 2,
          this.TILE_SIZE - 2,
          0x594100,
          0.2
        );
        tile.setStrokeStyle(1, 0xffb000, 0.5);
        tile.setInteractive({ useHandCursor: true });
        
        // Store grid position in tile data
        tile.setData('gridX', col);
        tile.setData('gridY', row);

        tile.on('pointerover', () => {
          if (this.currentMode === 'idle') {
            tile.setFillStyle(0xffb000, 0.3);
          }
        });

        tile.on('pointerout', () => {
          if (this.currentMode === 'idle') {
            tile.setFillStyle(0x594100, 0.2);
          }
        });
        
        tile.on('pointerdown', () => {
          this.handleTileClick(col, row);
        });

        this.grid[row][col] = tile;
      }
    }
  }

  private createPlayer(gridX: number, gridY: number): void {
    this.playerGridX = gridX;
    this.playerGridY = gridY;
    
    const tile = this.grid[gridY][gridX];
    this.playerSprite = this.add.circle(tile.x, tile.y, 18, 0x00ff00);
    this.playerSprite.setStrokeStyle(2, 0xffffff);

    this.playerText = this.add.text(tile.x, tile.y, 'P', {
      fontFamily: 'VT323',
      fontSize: '24px',
      color: '#000000',
    });
    this.playerText.setOrigin(0.5);
  }

  private createEnemy(gridX: number, gridY: number): void {
    const tile = this.grid[gridY][gridX];
    const enemySprite = this.add.circle(tile.x, tile.y, 18, 0xb026ff);
    enemySprite.setStrokeStyle(2, 0xffffff);

    const enemyText = this.add.text(tile.x, tile.y, 'E', {
      fontFamily: 'VT323',
      fontSize: '24px',
      color: '#FFFFFF',
    });
    enemyText.setOrigin(0.5);
    
    // Add enemy to tracking array
    this.enemies.push({
      sprite: enemySprite,
      text: enemyText,
      gridX: gridX,
      gridY: gridY,
      hp: 100,
      maxHp: 100
    });
  }

  private setupControlButtons(): void {
    const btnMove = document.getElementById('btn-move');
    const btnAttack = document.getElementById('btn-attack');
    const btnInventory = document.getElementById('btn-inventory');
    const btnEndTurn = document.getElementById('btn-end-turn');

    if (btnMove) {
      btnMove.onclick = () => this.handleMove();
    }

    if (btnAttack) {
      btnAttack.onclick = () => this.handleAttack();
    }

    if (btnInventory) {
      btnInventory.onclick = () => this.uiManager.toggleInventory();
    }

    if (btnEndTurn) {
      btnEndTurn.onclick = () => this.handleEndTurn();
    }
  }

  private startTurnTimer(): void {
    this.turnTimer = 30;
    this.isTimerActive = true;
    this.uiManager.updateTimer(this.turnTimer);

    this.timerEvent = this.time.addEvent({
      delay: 1000,
      callback: this.updateTimer,
      callbackScope: this,
      loop: true,
    });
  }

  private updateTimer(): void {
    if (!this.isTimerActive) return;

    this.turnTimer--;
    this.uiManager.updateTimer(this.turnTimer);

    if (this.turnTimer <= 5) {
      this.uiManager.setTimerWarning(true);
    }

    if (this.turnTimer <= 0) {
      this.handleTimeOut();
    }
  }

  private handleTimeOut(): void {
    this.isTimerActive = false;
    if (this.timerEvent) {
      this.timerEvent.remove();
    }

    // Check tray before timeout
    const trayItems = this.inventoryManager.getTray();
    const trayWarning = trayItems.length > 0 
      ? ` // ${trayItems.length} ITEM(S) LOST` 
      : '';
    
    this.uiManager.updateSystemMessage(`SYSTEM HALTED${trayWarning}`);
    
    // Trigger glitch effect
    const container = document.getElementById('game-container');
    if (container) {
      container.classList.add('glitch-active');
      
      setTimeout(() => {
        container.classList.remove('glitch-active');
        this.handleEndTurn();
      }, 800);
    }
  }

  private handleMove(): void {
    console.log('Move action');
    
    if (this.currentMode === 'moving') {
      // Cancel move mode
      this.clearHighlights();
      this.currentMode = 'idle';
      this.uiManager.updateSystemMessage('BATTLE MODE ACTIVE');
      return;
    }
    
    this.currentMode = 'moving';
    this.uiManager.updateSystemMessage('SELECT MOVEMENT TARGET');
    this.highlightMovementRange();
  }
  
  private highlightMovementRange(): void {
    // Clear previous highlights
    this.clearHighlights();
    
    // Highlight tiles within movement range
    for (let row = 0; row < this.GRID_SIZE; row++) {
      for (let col = 0; col < this.GRID_SIZE; col++) {
        const distance = Math.abs(col - this.playerGridX) + Math.abs(row - this.playerGridY);
        
        if (distance > 0 && distance <= this.MOVE_RANGE) {
          const tile = this.grid[row][col];
          tile.setFillStyle(0x00ff00, 0.3);
          tile.setStrokeStyle(2, 0x00ff00, 0.8);
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
  
  private handleTileClick(gridX: number, gridY: number): void {
    if (this.currentMode === 'moving') {
      const distance = Math.abs(gridX - this.playerGridX) + Math.abs(gridY - this.playerGridY);
      
      if (distance > 0 && distance <= this.MOVE_RANGE) {
        // Valid move
        this.movePlayerTo(gridX, gridY);
      } else {
        // Invalid move
        this.cameras.main.shake(50, 0.002);
        console.log('Out of range!');
      }
    } else if (this.currentMode === 'attacking') {
      const distance = Math.abs(gridX - this.playerGridX) + Math.abs(gridY - this.playerGridY);
      
      if (distance > 0 && distance <= this.ATTACK_RANGE) {
        // Valid attack range
        this.attackTarget(gridX, gridY);
      } else {
        // Out of attack range
        this.cameras.main.shake(50, 0.002);
        console.log('Out of attack range!');
      }
    }
  }
  
  private movePlayerTo(gridX: number, gridY: number): void {
    const targetTile = this.grid[gridY][gridX];
    
    // Calculate path for looting (simple straight line for now)
    const path = this.calculatePath(this.playerGridX, this.playerGridY, gridX, gridY);
    
    // Animate player movement
    this.tweens.add({
      targets: [this.playerSprite, this.playerText],
      x: targetTile.x,
      y: targetTile.y,
      duration: 300,
      ease: 'Power2',
      onComplete: () => {
        this.playerGridX = gridX;
        this.playerGridY = gridY;
        
        // Auto-loot items along the path
        path.forEach(pos => {
          this.collectItemsAt(pos.x, pos.y);
        });
        
        this.clearHighlights();
        this.currentMode = 'idle';
        
        // Check if we collected items
        const itemsAtDestination = this.fieldItems.filter(
          item => item.gridX === gridX && item.gridY === gridY
        );
        
        if (itemsAtDestination.length === 0) {
          this.uiManager.updateSystemMessage('MOVE COMPLETE');
        }
        
        console.log(`Player moved to (${gridX}, ${gridY})`);
      }
    });
  }

  /**
   * Calculate path from start to end (simple straight line)
   */
  private calculatePath(startX: number, startY: number, endX: number, endY: number): Array<{ x: number; y: number }> {
    const path: Array<{ x: number; y: number }> = [];
    
    // Simple Manhattan path (move horizontal first, then vertical)
    let currentX = startX;
    let currentY = startY;
    
    // Move horizontally
    while (currentX !== endX) {
      if (currentX < endX) {
        currentX++;
      } else {
        currentX--;
      }
      path.push({ x: currentX, y: currentY });
    }
    
    // Move vertically
    while (currentY !== endY) {
      if (currentY < endY) {
        currentY++;
      } else {
        currentY--;
      }
      path.push({ x: currentX, y: currentY });
    }
    
    return path;
  }
  
  private attackTarget(gridX: number, gridY: number): void {
    // Find enemy at target position
    const targetEnemy = this.enemies.find(e => e.gridX === gridX && e.gridY === gridY && e.hp > 0);
    
    if (!targetEnemy) {
      // No enemy at this position
      this.cameras.main.shake(50, 0.002);
      console.log('No target at this position!');
      return;
    }
    
    // Calculate damage
    const damage = 30 + Math.floor(Math.random() * 20); // 30-50 damage
    
    // Flash effect on player
    this.tweens.add({
      targets: this.playerSprite,
      alpha: 0.3,
      duration: 100,
      yoyo: true,
      repeat: 2
    });
    
    // Create attack line/projectile effect
    const attackLine = this.add.line(
      0, 0,
      this.playerSprite.x,
      this.playerSprite.y,
      targetEnemy.sprite.x,
      targetEnemy.sprite.y,
      0xffb000,
      1
    );
    attackLine.setLineWidth(3);
    attackLine.setOrigin(0, 0);
    
    // Fade out attack line
    this.tweens.add({
      targets: attackLine,
      alpha: 0,
      duration: 300,
      onComplete: () => {
        attackLine.destroy();
      }
    });
    
    // Damage enemy
    this.time.delayedCall(150, () => {
      targetEnemy.hp -= damage;
      
      // Shake camera
      this.cameras.main.shake(200, 0.01);
      
      // Flash enemy red
      targetEnemy.sprite.setFillStyle(0xff0000);
      
      this.time.delayedCall(200, () => {
        if (targetEnemy.hp <= 0) {
          // Enemy defeated
          this.uiManager.updateSystemMessage(`ENEMY ELIMINATED! (${damage} DMG)`);
          
          // Death animation
          this.tweens.add({
            targets: [targetEnemy.sprite, targetEnemy.text],
            alpha: 0,
            scale: 0.5,
            duration: 400,
            ease: 'Power2',
            onComplete: () => {
              targetEnemy.sprite.destroy();
              targetEnemy.text.destroy();
            }
          });
        } else {
          // Enemy still alive
          targetEnemy.sprite.setFillStyle(0xb026ff);
          this.uiManager.updateSystemMessage(`HIT! ${damage} DAMAGE (HP: ${targetEnemy.hp}/${targetEnemy.maxHp})`);
        }
      });
      
      // Clear highlights and reset mode
      this.clearHighlights();
      this.currentMode = 'idle';
    });
  }

  private handleAttack(): void {
    console.log('Attack action');
    
    if (this.currentMode === 'attacking') {
      // Cancel attack mode
      this.clearHighlights();
      this.currentMode = 'idle';
      this.uiManager.updateSystemMessage('BATTLE MODE ACTIVE');
      return;
    }
    
    this.currentMode = 'attacking';
    this.uiManager.updateSystemMessage('SELECT ATTACK TARGET');
    this.highlightAttackRange();
  }
  
  private highlightAttackRange(): void {
    // Clear previous highlights
    this.clearHighlights();
    
    // Highlight tiles within attack range
    for (let row = 0; row < this.GRID_SIZE; row++) {
      for (let col = 0; col < this.GRID_SIZE; col++) {
        const distance = Math.abs(col - this.playerGridX) + Math.abs(row - this.playerGridY);
        
        if (distance > 0 && distance <= this.ATTACK_RANGE) {
          const tile = this.grid[row][col];
          
          // Check if there's an enemy on this tile
          const hasEnemy = this.enemies.some(e => e.gridX === col && e.gridY === row && e.hp > 0);
          
          if (hasEnemy) {
            // Red highlight for enemy tiles
            tile.setFillStyle(0xff0000, 0.5);
            tile.setStrokeStyle(2, 0xff0000, 1);
          } else {
            // Violet highlight for empty tiles in range
            tile.setFillStyle(0xb026ff, 0.3);
            tile.setStrokeStyle(2, 0xb026ff, 0.8);
          }
          
          this.highlightedTiles.push(tile);
        }
      }
    }
  }

  private handleEndTurn(): void {
    console.log('Turn ended');
    
    if (this.timerEvent) {
      this.timerEvent.remove();
    }

    this.uiManager.setTimerWarning(false);
    
    // Check if there are items in tray that will be lost
    const trayItems = this.inventoryManager.getTray();
    if (trayItems.length > 0) {
      this.uiManager.updateSystemMessage(`WARNING: ${trayItems.length} ITEM(S) IN TRAY WILL BE LOST!`);
      
      // Clear tray after warning
      this.time.delayedCall(1500, () => {
        this.inventoryManager.clearTray();
        
        // Update inventory UI
        const inventoryUI = this.uiManager.getInventoryUI();
        if (inventoryUI) {
          inventoryUI.update();
        }
        
        this.uiManager.updateSystemMessage('ENEMY TURN');
        
        // Simulate enemy turn
        this.time.delayedCall(2000, () => {
          this.uiManager.updateSystemMessage('YOUR TURN');
          this.startTurnTimer();
        });
      });
    } else {
      this.uiManager.updateSystemMessage('ENEMY TURN');
      
      // Simulate enemy turn
      this.time.delayedCall(2000, () => {
        this.uiManager.updateSystemMessage('YOUR TURN');
        this.startTurnTimer();
      });
    }
  }

  /**
   * Create an item on the map at specified grid position
   */
  private createItemOnMap(gridX: number, gridY: number, itemData: ItemData): void {
    const tile = this.grid[gridY][gridX];
    
    // Create container for item sprite
    const container = this.add.container(tile.x, tile.y);
    
    // Create treasure chest sprite (placeholder)
    const chest = this.add.rectangle(0, 0, 30, 30, 0xffb000);
    chest.setStrokeStyle(2, 0x594100);
    
    // Add glow effect
    const glow = this.add.circle(0, 0, 20, 0xffb000, 0.3);
    
    // Add item icon/text
    const itemIcon = this.add.text(0, 0, 'I', {
      fontFamily: 'VT323',
      fontSize: '20px',
      color: '#000000',
    });
    itemIcon.setOrigin(0.5);
    
    container.add([glow, chest, itemIcon]);
    
    // Pulse animation
    this.tweens.add({
      targets: glow,
      scale: 1.2,
      alpha: 0.5,
      duration: 1000,
      yoyo: true,
      repeat: -1,
      ease: 'Sine.easeInOut',
    });
    
    // Add to field items tracking
    this.fieldItems.push({
      sprite: container,
      gridX,
      gridY,
      itemData,
    });
    
    // Make tile glow
    tile.setFillStyle(0xffb000, 0.3);
  }

  /**
   * Check and collect items on a specific tile
   */
  private collectItemsAt(gridX: number, gridY: number): void {
    // Find items at this position
    const itemsAtPosition = this.fieldItems.filter(
      item => item.gridX === gridX && item.gridY === gridY
    );
    
    if (itemsAtPosition.length === 0) return;
    
    itemsAtPosition.forEach(fieldItem => {
      // Add item to inventory tray
      this.inventoryManager.addItemToTray(fieldItem.itemData);
      
      // Show floating text
      this.showFloatingText(
        fieldItem.sprite.x,
        fieldItem.sprite.y,
        `+${fieldItem.itemData.name}`
      );
      
      // Destroy sprite with animation
      this.tweens.add({
        targets: fieldItem.sprite,
        scale: 1.5,
        alpha: 0,
        duration: 300,
        ease: 'Power2',
        onComplete: () => {
          fieldItem.sprite.destroy();
        },
      });
      
      // Remove from tracking array
      const index = this.fieldItems.indexOf(fieldItem);
      if (index > -1) {
        this.fieldItems.splice(index, 1);
      }
      
      // Reset tile appearance
      const tile = this.grid[gridY][gridX];
      tile.setFillStyle(0x594100, 0.2);
    });
    
    // Update inventory UI
    const inventoryUI = this.uiManager.getInventoryUI();
    if (inventoryUI) {
      inventoryUI.update();
    }
    
    // Update system message
    if (itemsAtPosition.length === 1) {
      this.uiManager.updateSystemMessage(`ACQUIRED: ${itemsAtPosition[0].itemData.name}`);
    } else {
      this.uiManager.updateSystemMessage(`ACQUIRED ${itemsAtPosition.length} ITEMS`);
    }
  }

  /**
   * Show floating text effect
   */
  private showFloatingText(x: number, y: number, text: string): void {
    const floatingText = this.add.text(x, y, text, {
      fontFamily: 'VT323',
      fontSize: '20px',
      color: '#FFB000',
      stroke: '#000000',
      strokeThickness: 2,
    });
    floatingText.setOrigin(0.5);
    
    this.tweens.add({
      targets: floatingText,
      y: y - 50,
      alpha: 0,
      duration: 1500,
      ease: 'Power2',
      onComplete: () => {
        floatingText.destroy();
      },
    });
  }

  /**
   * Create test items for demonstration
   */
  private createTestItems(): void {
    // Create sample weapon item
    const weaponItem: ItemData = {
      id: `weapon-${Date.now()}`,
      name: 'PLASMA GUN',
      type: ItemType.WEAPON,
      shape: [
        [1, 1],
        [1, 0],
      ],
      rotation: 0,
      maxAmmo: 30,
      currentAmmo: 30,
      scrapValue: 15,
      description: 'A basic plasma weapon',
    };
    
    // Create sample consumable item
    const consumableItem: ItemData = {
      id: `consumable-${Date.now() + 1}`,
      name: 'MED KIT',
      type: ItemType.CONSUMABLE,
      shape: [[1]],
      rotation: 0,
      maxAmmo: null,
      currentAmmo: null,
      scrapValue: 5,
      description: 'Restores health',
    };
    
    // Create sample armor item
    const armorItem: ItemData = {
      id: `armor-${Date.now() + 2}`,
      name: 'SHIELD CORE',
      type: ItemType.ARMOR,
      shape: [
        [1, 1],
        [1, 1],
      ],
      rotation: 0,
      maxAmmo: null,
      currentAmmo: null,
      scrapValue: 20,
      description: 'Provides protection',
    };
    
    // Place items on map
    this.createItemOnMap(4, 3, weaponItem);
    this.createItemOnMap(5, 5, consumableItem);
    this.createItemOnMap(1, 6, armorItem);
  }

  update(): void {
    // Game loop updates here
  }
}
