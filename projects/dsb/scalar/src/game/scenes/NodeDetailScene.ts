import Phaser from 'phaser';

import { GameStateManager } from '../systems/GameState';
import { NodeType } from '../types';

interface SceneData {
  nodeId: string;
}

export class NodeDetailScene extends Phaser.Scene {
  private gameState: GameStateManager;
  private nodeId!: string;
  private background!: Phaser.GameObjects.Graphics;
  private contentContainer!: Phaser.GameObjects.Container;
  private uiContainer!: Phaser.GameObjects.Container;

  constructor() {
    super({ key: 'NodeDetailScene' });
    this.gameState = new GameStateManager();
  }

  init(data: SceneData): void {
    this.nodeId = data.nodeId;
  }

  create(): void {
    this.createBackground();
    this.createNodeContent();
    this.createUI();
    this.createAnimations();
  }

  private createBackground(): void {
    // Create background based on node type
    const node = this.gameState.getNodes().get(this.nodeId);
    if (!node) return;

    this.background = this.add.graphics();
    
    // Different backgrounds for different node types
    switch (node.type) {
      case NodeType.START:
        this.background.fillGradientStyle(0x4CAF50, 0x2E7D32, 0x388E3C, 0x1B5E20, 1);
        break;
      case NodeType.EXPLORATION:
        this.background.fillGradientStyle(0x2196F3, 0x1565C0, 0x1976D2, 0x0D47A1, 1);
        break;
      case NodeType.RESOURCE:
        this.background.fillGradientStyle(0xFF9800, 0xF57C00, 0xFFB74D, 0xE65100, 1);
        break;
      case NodeType.ENEMY:
        this.background.fillGradientStyle(0xF44336, 0xD32F2F, 0xEF5350, 0xC62828, 1);
        break;
      case NodeType.TREASURE:
        this.background.fillGradientStyle(0xFFD700, 0xFFC107, 0xFFEB3B, 0xFF8F00, 1);
        break;
      case NodeType.BOSS:
        this.background.fillGradientStyle(0x9C27B0, 0x7B1FA2, 0xBA68C8, 0x6A1B9A, 1);
        break;
      case NodeType.EXIT:
        this.background.fillGradientStyle(0x00BCD4, 0x0097A7, 0x4DD0E1, 0x00695C, 1);
        break;
      default:
        this.background.fillGradientStyle(0x1a1a2e, 0x16213e, 0x0f3460, 0x533483, 1);
    }
    
    this.background.fillRect(0, 0, this.scale.width, this.scale.height);
  }

  private createNodeContent(): void {
    const node = this.gameState.getNodes().get(this.nodeId);
    if (!node) return;

    this.contentContainer = this.add.container(this.scale.width / 2, this.scale.height / 2);

    // Node title
    const title = this.add.text(0, -300, node.name, {
      fontSize: '36px',
      fontFamily: 'Arial, sans-serif',
      color: '#ffffff',
      fontStyle: 'bold'
    });
    title.setOrigin(0.5);

    // Node description
    const description = this.add.text(0, -250, node.description, {
      fontSize: '18px',
      fontFamily: 'Arial, sans-serif',
      color: '#ffffff',
      wordWrap: { width: 600 }
    });
    description.setOrigin(0.5);

    // Create content based on node type
    switch (node.type) {
      case NodeType.RESOURCE:
        this.createResourceContent(node);
        break;
      case NodeType.ENEMY:
        this.createEnemyContent(node);
        break;
      case NodeType.TREASURE:
        this.createTreasureContent(node);
        break;
      case NodeType.BOSS:
        this.createBossContent(node);
        break;
      default:
        this.createExplorationContent(node);
    }

    this.contentContainer.add([title, description]);
  }

  private createResourceContent(node: any): void {
    if (!node.resources) return;

    const resourceTitle = this.add.text(0, -150, 'Available Resources:', {
      fontSize: '24px',
      fontFamily: 'Arial, sans-serif',
      color: '#ffffff',
      fontStyle: 'bold'
    });
    resourceTitle.setOrigin(0.5);

    let yOffset = -100;
    node.resources.forEach((resource: any, _index: number) => {
      const resourceText = this.add.text(0, yOffset, `${resource.name}: ${resource.amount}/${resource.maxAmount}`, {
        fontSize: '18px',
        fontFamily: 'Arial, sans-serif',
        color: this.getRarityColor(resource.rarity)
      });
      resourceText.setOrigin(0.5);

      const collectButton = this.add.text(200, yOffset, 'COLLECT', {
        fontSize: '16px',
        fontFamily: 'Arial, sans-serif',
        color: '#ffffff',
        backgroundColor: '#4CAF50',
        padding: { x: 10, y: 5 }
      });
      collectButton.setOrigin(0.5);
      collectButton.setInteractive({ useHandCursor: true });
      collectButton.on('pointerdown', () => {
        this.collectResource(resource);
      });

      this.contentContainer.add([resourceText, collectButton]);
      yOffset += 40;
    });

    this.contentContainer.add(resourceTitle);
  }

  private createEnemyContent(node: any): void {
    if (!node.enemies) return;

    const enemyTitle = this.add.text(0, -150, 'Enemies:', {
      fontSize: '24px',
      fontFamily: 'Arial, sans-serif',
      color: '#ffffff',
      fontStyle: 'bold'
    });
    enemyTitle.setOrigin(0.5);

    let yOffset = -100;
    node.enemies.forEach((enemy: any, _index: number) => {
      const enemyText = this.add.text(0, yOffset, `${enemy.name} (Level ${enemy.level})`, {
        fontSize: '18px',
        fontFamily: 'Arial, sans-serif',
        color: '#ffffff'
      });
      enemyText.setOrigin(0.5);

      const statsText = this.add.text(0, yOffset + 25, `HP: ${enemy.health}/${enemy.maxHealth} | ATK: ${enemy.attack} | DEF: ${enemy.defense}`, {
        fontSize: '14px',
        fontFamily: 'Arial, sans-serif',
        color: '#cccccc'
      });
      statsText.setOrigin(0.5);

      const fightButton = this.add.text(200, yOffset, 'FIGHT', {
        fontSize: '16px',
        fontFamily: 'Arial, sans-serif',
        color: '#ffffff',
        backgroundColor: '#F44336',
        padding: { x: 10, y: 5 }
      });
      fightButton.setOrigin(0.5);
      fightButton.setInteractive({ useHandCursor: true });
      fightButton.on('pointerdown', () => {
        this.startCombat(enemy);
      });

      this.contentContainer.add([enemyText, statsText, fightButton]);
      yOffset += 80;
    });

    this.contentContainer.add(enemyTitle);
  }

  private createTreasureContent(node: any): void {
    const treasureText = this.add.text(0, -100, 'A mysterious treasure awaits...', {
      fontSize: '24px',
      fontFamily: 'Arial, sans-serif',
      color: '#FFD700',
      fontStyle: 'bold'
    });
    treasureText.setOrigin(0.5);

    if (node.requirements) {
      const requirementsText = this.add.text(0, -50, 'Requirements:', {
        fontSize: '20px',
        fontFamily: 'Arial, sans-serif',
        color: '#ffffff'
      });
      requirementsText.setOrigin(0.5);

      let yOffset = -20;
      node.requirements.forEach((req: string) => {
        const reqText = this.add.text(0, yOffset, `• ${req}`, {
          fontSize: '16px',
          fontFamily: 'Arial, sans-serif',
          color: '#cccccc'
        });
        reqText.setOrigin(0.5);
        this.contentContainer.add(reqText);
        yOffset += 25;
      });

      this.contentContainer.add(requirementsText);
    }

    const openButton = this.add.text(0, 50, 'OPEN TREASURE', {
      fontSize: '20px',
      fontFamily: 'Arial, sans-serif',
      color: '#ffffff',
      backgroundColor: '#FFD700',
      padding: { x: 15, y: 8 }
    });
    openButton.setOrigin(0.5);
    openButton.setInteractive({ useHandCursor: true });
    openButton.on('pointerdown', () => {
      this.openTreasure(node);
    });

    this.contentContainer.add([treasureText, openButton]);
  }

  private createBossContent(node: any): void {
    if (!node.enemies) return;

    const boss = node.enemies[0];
    const bossTitle = this.add.text(0, -200, 'BOSS ENCOUNTER', {
      fontSize: '32px',
      fontFamily: 'Arial, sans-serif',
      color: '#FF0000',
      fontStyle: 'bold'
    });
    bossTitle.setOrigin(0.5);

    const bossName = this.add.text(0, -150, boss.name, {
      fontSize: '28px',
      fontFamily: 'Arial, sans-serif',
      color: '#ffffff',
      fontStyle: 'bold'
    });
    bossName.setOrigin(0.5);

    const bossStats = this.add.text(0, -100, `Level ${boss.level} | HP: ${boss.health}/${boss.maxHealth}`, {
      fontSize: '20px',
      fontFamily: 'Arial, sans-serif',
      color: '#cccccc'
    });
    bossStats.setOrigin(0.5);

    const bossDescription = this.add.text(0, -50, 'This is a powerful foe. Are you ready for the challenge?', {
      fontSize: '18px',
      fontFamily: 'Arial, sans-serif',
      color: '#ffffff',
      wordWrap: { width: 500 }
    });
    bossDescription.setOrigin(0.5);

    const fightButton = this.add.text(0, 50, 'CHALLENGE BOSS', {
      fontSize: '24px',
      fontFamily: 'Arial, sans-serif',
      color: '#ffffff',
      backgroundColor: '#9C27B0',
      padding: { x: 20, y: 10 }
    });
    fightButton.setOrigin(0.5);
    fightButton.setInteractive({ useHandCursor: true });
    fightButton.on('pointerdown', () => {
      this.startBossFight(boss);
    });

    this.contentContainer.add([bossTitle, bossName, bossStats, bossDescription, fightButton]);
  }

  private createExplorationContent(node: any): void {
    const exploreText = this.add.text(0, -100, 'This area is ready for exploration...', {
      fontSize: '24px',
      fontFamily: 'Arial, sans-serif',
      color: '#ffffff'
    });
    exploreText.setOrigin(0.5);

    const exploreButton = this.add.text(0, 0, 'EXPLORE', {
      fontSize: '20px',
      fontFamily: 'Arial, sans-serif',
      color: '#ffffff',
      backgroundColor: '#2196F3',
      padding: { x: 15, y: 8 }
    });
    exploreButton.setOrigin(0.5);
    exploreButton.setInteractive({ useHandCursor: true });
    exploreButton.on('pointerdown', () => {
      this.exploreArea(node);
    });

    this.contentContainer.add([exploreText, exploreButton]);
  }

  private createUI(): void {
    this.uiContainer = this.add.container(0, 0);

    // Back button
    const backButton = this.add.text(100, 50, '← BACK TO MAP', {
      fontSize: '18px',
      fontFamily: 'Arial, sans-serif',
      color: '#ffffff',
      backgroundColor: '#666666',
      padding: { x: 10, y: 5 }
    });
    backButton.setInteractive({ useHandCursor: true });
    backButton.on('pointerdown', () => {
      this.scene.start('ExplorationScene');
    });

    this.uiContainer.add(backButton);
  }

  private createAnimations(): void {
    // TODO: Implement particle system when Phaser particles API is available
  }

  private getRarityColor(rarity: string): string {
    switch (rarity) {
      case 'common': return '#ffffff';
      case 'uncommon': return '#00ff00';
      case 'rare': return '#0080ff';
      case 'epic': return '#8000ff';
      case 'legendary': return '#ff8000';
      default: return '#ffffff';
    }
  }

  private collectResource(resource: any): void {
    const amount = Math.min(resource.amount, 3); // Collect up to 3 at a time
    this.gameState.addResource(resource.id, amount);
    resource.amount = Math.max(0, resource.amount - amount);
    
    this.showMessage(`Collected ${amount} ${resource.name}!`);
    
    // Update display
    this.scene.restart({ nodeId: this.nodeId });
  }

  private startCombat(enemy: any): void {
    this.showMessage(`Combat with ${enemy.name} started!`);
    // TODO: Implement combat system
    this.gameState.completeNode(this.nodeId);
    this.scene.start('ExplorationScene');
  }

  private startBossFight(boss: any): void {
    this.showMessage(`Boss fight with ${boss.name} started!`);
    // TODO: Implement boss combat system
    this.gameState.completeNode(this.nodeId);
    this.scene.start('ExplorationScene');
  }

  private openTreasure(node: any): void {
    if (node.requirements) {
      const canOpen = node.requirements.every((req: string) => 
        this.gameState.getPlayer().resources.has(req) && 
        this.gameState.getPlayer().resources.get(req)! > 0
      );
      
      if (!canOpen) {
        this.showMessage('You do not have the required items to open this treasure.');
        return;
      }
    }
    
    // Give random rewards
    const rewards = ['gold', 'experience', 'rare_item'];
    const reward = rewards[Math.floor(Math.random() * rewards.length)];
    
    switch (reward) {
      case 'gold':
        this.gameState.addResource('gold', 100);
        this.showMessage('You found 100 gold!');
        break;
      case 'experience':
        this.gameState.addExperience(50);
        this.showMessage('You gained 50 experience!');
        break;
      case 'rare_item':
        this.gameState.addResource('rare_item', 1);
        this.showMessage('You found a rare item!');
        break;
    }
    
    this.gameState.completeNode(this.nodeId);
    this.scene.start('ExplorationScene');
  }

  private exploreArea(_node: any): void {
    this.gameState.addExperience(10);
    this.showMessage('You explored the area and gained 10 experience!');
    this.gameState.completeNode(this.nodeId);
    this.scene.start('ExplorationScene');
  }

  private showMessage(message: string): void {
    const text = this.add.text(this.scale.width / 2, this.scale.height - 100, message, {
      fontSize: '18px',
      fontFamily: 'Arial, sans-serif',
      color: '#ffffff',
      backgroundColor: '#000000',
      padding: { x: 10, y: 5 }
    });
    text.setOrigin(0.5);
    
    this.tweens.add({
      targets: text,
      alpha: 0,
      duration: 3000,
      onComplete: () => {
        text.destroy();
      }
    });
  }
} 