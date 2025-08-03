import Phaser from 'phaser';

import { GameStateManager } from '../systems/GameState';
import { NodeState } from '../types';

export class ExplorationScene extends Phaser.Scene {
  private gameState: GameStateManager;
  private nodeSprites: Map<string, Phaser.GameObjects.Container> = new Map();
  private connectionLines!: Phaser.GameObjects.Graphics;
  private playerSprite!: Phaser.GameObjects.Sprite;
  private uiContainer!: Phaser.GameObjects.Container;
  private camera!: Phaser.Cameras.Scene2D.Camera;

  constructor() {
    super({ key: 'ExplorationScene' });
    this.gameState = new GameStateManager();
  }

  create(): void {
    this.createBackground();
    this.createNodeTextures();
    this.createNodes();
    this.createConnections();
    this.createPlayer();
    this.createUI();
    this.setupCamera();
    this.centerOnCurrentNode();
  }

  private createBackground(): void {
    // Create gradient background
    const graphics = this.add.graphics();
    graphics.fillGradientStyle(0x0f1419, 0x1a1a2e, 0x16213e, 0x0f3460, 1);
    graphics.fillRect(0, 0, this.scale.width, this.scale.height);

    // Add grid pattern
    graphics.lineStyle(1, 0xffffff, 0.1);
    for (let x = 0; x < this.scale.width; x += 50) {
      graphics.moveTo(x, 0);
      graphics.lineTo(x, this.scale.height);
    }
    for (let y = 0; y < this.scale.height; y += 50) {
      graphics.moveTo(0, y);
      graphics.lineTo(this.scale.width, y);
    }
    graphics.strokePath();
  }

  private createNodeTextures(): void {
    // Create textures for different node types
    const nodeTypes = [
      { key: 'start', color: 0x4CAF50, borderColor: 0x2E7D32 },
      { key: 'exploration', color: 0x2196F3, borderColor: 0x1565C0 },
      { key: 'resource', color: 0xFF9800, borderColor: 0xF57C00 },
      { key: 'enemy', color: 0xF44336, borderColor: 0xD32F2F },
      { key: 'treasure', color: 0xFFD700, borderColor: 0xFFC107 },
      { key: 'boss', color: 0x9C27B0, borderColor: 0x7B1FA2 },
      { key: 'exit', color: 0x00BCD4, borderColor: 0x0097A7 }
    ];

    nodeTypes.forEach(({ key, color, borderColor }) => {
      const graphics = this.add.graphics();
      graphics.fillStyle(color);
      graphics.fillCircle(25, 25, 25);
      graphics.lineStyle(3, borderColor);
      graphics.strokeCircle(25, 25, 25);
      graphics.generateTexture(`node_${key}`, 50, 50);
      graphics.destroy();
    });

    // Create locked node texture
    const lockedGraphics = this.add.graphics();
    lockedGraphics.fillStyle(0x666666);
    lockedGraphics.fillCircle(25, 25, 25);
    lockedGraphics.lineStyle(3, 0x444444);
    lockedGraphics.strokeCircle(25, 25, 25);
    lockedGraphics.lineStyle(2, 0x444444);
    lockedGraphics.strokeRect(15, 15, 20, 20);
    lockedGraphics.generateTexture('node_locked', 50, 50);
    lockedGraphics.destroy();
  }

  private createNodes(): void {
    const nodes = this.gameState.getNodes();
    
    nodes.forEach((nodeData, nodeId) => {
      const container = this.add.container(nodeData.x, nodeData.y);
      
      // Determine texture based on node state and type
      let textureKey = 'node_locked';
      if (nodeData.state !== NodeState.LOCKED) {
        textureKey = `node_${nodeData.type}`;
      }
      
      const sprite = this.add.sprite(0, 0, textureKey);
      sprite.setInteractive({ useHandCursor: true });
      
      // Add node name
      const nameText = this.add.text(0, 40, nodeData.name, {
        fontSize: '14px',
        fontFamily: 'Arial, sans-serif',
        color: '#ffffff',
        fontStyle: 'bold'
      });
      nameText.setOrigin(0.5);
      
      // Add state indicator
      const stateText = this.add.text(0, 55, this.getStateText(nodeData.state), {
        fontSize: '10px',
        fontFamily: 'Arial, sans-serif',
        color: this.getStateColor(nodeData.state)
      });
      stateText.setOrigin(0.5);
      
      container.add([sprite, nameText, stateText]);
      
      // Add click handler
      sprite.on('pointerdown', () => {
        this.onNodeClick(nodeId);
      });
      
      // Add hover effects
      sprite.on('pointerover', () => {
        sprite.setScale(1.1);
        this.showNodeTooltip(nodeData);
      });
      
      sprite.on('pointerout', () => {
        sprite.setScale(1);
        this.hideNodeTooltip();
      });
      
      this.nodeSprites.set(nodeId, container);
    });
  }

  private createConnections(): void {
    this.connectionLines = this.add.graphics();
    this.connectionLines.lineStyle(2, 0x4a90e2, 0.6);
    
    const nodes = this.gameState.getNodes();
    nodes.forEach((nodeData) => {
      nodeData.connections.forEach(connectionId => {
        const connectedNode = nodes.get(connectionId);
        if (connectedNode) {
          this.connectionLines.moveTo(nodeData.x, nodeData.y);
          this.connectionLines.lineTo(connectedNode.x, connectedNode.y);
        }
      });
    });
    
    this.connectionLines.strokePath();
  }

  private createPlayer(): void {
    // Create player sprite
    const graphics = this.add.graphics();
    graphics.fillStyle(0x00ff00);
    graphics.fillCircle(12, 12, 12);
    graphics.lineStyle(2, 0x00cc00);
    graphics.strokeCircle(12, 12, 12);
    graphics.generateTexture('player', 24, 24);
    graphics.destroy();
    
    const currentNode = this.gameState.getCurrentNode();
    if (currentNode) {
      this.playerSprite = this.add.sprite(currentNode.x, currentNode.y, 'player');
    }
  }

  private createUI(): void {
    this.uiContainer = this.add.container(0, 0);
    
    // Player stats panel
    const statsPanel = this.add.graphics();
    statsPanel.fillStyle(0x000000, 0.8);
    statsPanel.fillRoundedRect(10, 10, 300, 100, 10);
    statsPanel.lineStyle(2, 0x4a90e2);
    statsPanel.strokeRoundedRect(10, 10, 300, 100, 10);
    
    const player = this.gameState.getPlayer();
    
    const levelText = this.add.text(20, 20, `Level: ${player.level}`, {
      fontSize: '16px',
      fontFamily: 'Arial, sans-serif',
      color: '#ffffff'
    });
    
    const healthText = this.add.text(20, 40, `Health: ${player.health}/${player.maxHealth}`, {
      fontSize: '16px',
      fontFamily: 'Arial, sans-serif',
      color: '#ffffff'
    });
    
    const expText = this.add.text(20, 60, `Experience: ${player.experience}`, {
      fontSize: '16px',
      fontFamily: 'Arial, sans-serif',
      color: '#ffffff'
    });
    
    const resourcesText = this.add.text(20, 80, `Resources: ${player.resources.size}`, {
      fontSize: '16px',
      fontFamily: 'Arial, sans-serif',
      color: '#ffffff'
    });
    
    // Menu button
    const menuButton = this.add.text(this.scale.width - 100, 20, 'MENU', {
      fontSize: '18px',
      fontFamily: 'Arial, sans-serif',
      color: '#ffffff',
      backgroundColor: '#4a90e2',
      padding: { x: 10, y: 5 }
    });
    menuButton.setInteractive({ useHandCursor: true });
    menuButton.on('pointerdown', () => {
      this.scene.start('MainMenuScene');
    });
    
    this.uiContainer.add([statsPanel, levelText, healthText, expText, resourcesText, menuButton]);
  }

  private setupCamera(): void {
    this.camera = this.cameras.main;
    this.camera.setBounds(0, 0, 1200, 800);
  }

  private centerOnCurrentNode(): void {
    const currentNode = this.gameState.getCurrentNode();
    if (currentNode) {
      this.camera.centerOn(currentNode.x, currentNode.y);
    }
  }

  private onNodeClick(nodeId: string): void {
    const node = this.gameState.getNodes().get(nodeId);
    if (!node) return;
    
    // Check if node is accessible
    if (!this.gameState.canAccessNode(nodeId)) {
      this.showMessage('This node is locked. You need to meet the requirements to access it.');
      return;
    }
    
    // Check if node is connected to current node
    const currentNode = this.gameState.getCurrentNode();
    if (currentNode && !currentNode.connections.includes(nodeId)) {
      this.showMessage('This node is not directly connected to your current location.');
      return;
    }
    
    // Move to node
    this.moveToNode(nodeId);
  }

  private moveToNode(nodeId: string): void {
    const targetNode = this.gameState.getNodes().get(nodeId);
    if (!targetNode) return;
    
    // Animate player movement
    this.tweens.add({
      targets: this.playerSprite,
      x: targetNode.x,
      y: targetNode.y,
      duration: 1000,
      ease: 'Power2',
      onComplete: () => {
        this.gameState.setCurrentNode(nodeId);
        this.gameState.exploreNode(nodeId);
        
        // Update node appearance
        this.updateNodeAppearance(nodeId);
        
        // Show node details
        this.scene.start('NodeDetailScene', { nodeId });
      }
    });
  }

  private updateNodeAppearance(nodeId: string): void {
    const container = this.nodeSprites.get(nodeId);
    if (!container) return;
    
    const node = this.gameState.getNodes().get(nodeId);
    if (!node) return;
    
    const sprite = container.getAt(0) as Phaser.GameObjects.Sprite;
    const stateText = container.getAt(2) as Phaser.GameObjects.Text;
    
    // Update texture if needed
    if (node.state !== NodeState.LOCKED) {
      sprite.setTexture(`node_${node.type}`);
    }
    
    // Update state text
    stateText.setText(this.getStateText(node.state));
    stateText.setColor(this.getStateColor(node.state));
  }

  private getStateText(state: NodeState): string {
    switch (state) {
      case NodeState.LOCKED: return 'LOCKED';
      case NodeState.UNLOCKED: return 'UNLOCKED';
      case NodeState.EXPLORED: return 'EXPLORED';
      case NodeState.COMPLETED: return 'COMPLETED';
      default: return '';
    }
  }

  private getStateColor(state: NodeState): string {
    switch (state) {
      case NodeState.LOCKED: return '#ff4444';
      case NodeState.UNLOCKED: return '#ffaa00';
      case NodeState.EXPLORED: return '#44ff44';
      case NodeState.COMPLETED: return '#4444ff';
      default: return '#ffffff';
    }
  }

  private showNodeTooltip(_nodeData: any): void {
    // TODO: Implement tooltip system
  }

  private hideNodeTooltip(): void {
    // TODO: Implement tooltip system
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