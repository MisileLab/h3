import { NodeData, NodeType, NodeState, PlayerData, GameState as IGameState } from '../types';

export class GameStateManager {
  private gameState: IGameState;

  constructor() {
    this.gameState = this.initializeGameState();
  }

  private initializeGameState(): IGameState {
    const player: PlayerData = {
      level: 1,
      experience: 0,
      health: 100,
      maxHealth: 100,
      resources: new Map(),
      exploredNodes: new Set(),
      currentNode: 'start',
      inventory: []
    };

    const nodes = this.generateNodeMap();

    return {
      player,
      nodes,
      currentNode: 'start',
      gamePhase: 'menu'
    };
  }

  private generateNodeMap(): Map<string, NodeData> {
    const nodes = new Map<string, NodeData>();

    // Start node
    nodes.set('start', {
      id: 'start',
      x: 600,
      y: 400,
      type: NodeType.START,
      state: NodeState.EXPLORED,
      connections: ['node_1', 'node_2'],
      name: 'Starting Point',
      description: 'Your journey begins here. Choose your path wisely.'
    });

    // Exploration nodes
    nodes.set('node_1', {
      id: 'node_1',
      x: 400,
      y: 300,
      type: NodeType.EXPLORATION,
      state: NodeState.UNLOCKED,
      connections: ['start', 'node_3', 'resource_1'],
      name: 'Ancient Forest',
      description: 'A mysterious forest with ancient trees and hidden secrets.'
    });

    nodes.set('node_2', {
      id: 'node_2',
      x: 800,
      y: 300,
      type: NodeType.EXPLORATION,
      state: NodeState.UNLOCKED,
      connections: ['start', 'enemy_1', 'node_4'],
      name: 'Dark Cave',
      description: 'A foreboding cave that echoes with unknown sounds.'
    });

    // Resource nodes
    nodes.set('resource_1', {
      id: 'resource_1',
      x: 300,
      y: 200,
      type: NodeType.RESOURCE,
      state: NodeState.LOCKED,
      connections: ['node_1'],
      name: 'Crystal Mine',
      description: 'Rich deposits of valuable crystals.',
      resources: [
        {
          id: 'crystal',
          name: 'Mystic Crystal',
          amount: 5,
          maxAmount: 10,
          rarity: 'rare'
        }
      ]
    });

    // Enemy nodes
    nodes.set('enemy_1', {
      id: 'enemy_1',
      x: 900,
      y: 200,
      type: NodeType.ENEMY,
      state: NodeState.LOCKED,
      connections: ['node_2'],
      name: 'Goblin Camp',
      description: 'A group of hostile goblins guards this area.',
      enemies: [
        {
          id: 'goblin_warrior',
          name: 'Goblin Warrior',
          health: 50,
          maxHealth: 50,
          attack: 15,
          defense: 5,
          level: 2
        }
      ]
    });

    // Treasure nodes
    nodes.set('treasure_1', {
      id: 'treasure_1',
      x: 200,
      y: 500,
      type: NodeType.TREASURE,
      state: NodeState.LOCKED,
      connections: ['node_3'],
      name: 'Hidden Vault',
      description: 'A secret vault containing ancient treasures.',
      requirements: ['key_fragment_1', 'key_fragment_2']
    });

    // Boss node
    nodes.set('boss_1', {
      id: 'boss_1',
      x: 600,
      y: 100,
      type: NodeType.BOSS,
      state: NodeState.LOCKED,
      connections: ['node_4'],
      name: 'Dragon\'s Lair',
      description: 'The lair of an ancient dragon. Only the strongest may enter.',
      enemies: [
        {
          id: 'ancient_dragon',
          name: 'Ancient Dragon',
          health: 200,
          maxHealth: 200,
          attack: 40,
          defense: 20,
          level: 10
        }
      ],
      requirements: ['dragon_key']
    });

    // Additional exploration nodes
    nodes.set('node_3', {
      id: 'node_3',
      x: 400,
      y: 500,
      type: NodeType.EXPLORATION,
      state: NodeState.LOCKED,
      connections: ['node_1', 'treasure_1', 'node_5'],
      name: 'Abandoned Village',
      description: 'An eerie village with no signs of life.'
    });

    nodes.set('node_4', {
      id: 'node_4',
      x: 800,
      y: 500,
      type: NodeType.EXPLORATION,
      state: NodeState.LOCKED,
      connections: ['node_2', 'boss_1', 'node_5'],
      name: 'Mountain Pass',
      description: 'A treacherous mountain path with stunning views.'
    });

    nodes.set('node_5', {
      id: 'node_5',
      x: 600,
      y: 600,
      type: NodeType.EXIT,
      state: NodeState.LOCKED,
      connections: ['node_3', 'node_4'],
      name: 'Portal to Next Realm',
      description: 'A mysterious portal that leads to the next realm.',
      requirements: ['realm_key']
    });

    return nodes;
  }

  getGameState(): IGameState {
    return this.gameState;
  }

  getPlayer(): PlayerData {
    return this.gameState.player;
  }

  getNodes(): Map<string, NodeData> {
    return this.gameState.nodes;
  }

  getCurrentNode(): NodeData | undefined {
    return this.gameState.nodes.get(this.gameState.currentNode);
  }

  setCurrentNode(nodeId: string): void {
    if (this.gameState.nodes.has(nodeId)) {
      this.gameState.currentNode = nodeId;
      this.gameState.player.currentNode = nodeId;
    }
  }

  exploreNode(nodeId: string): void {
    const node = this.gameState.nodes.get(nodeId);
    if (node && node.state === NodeState.UNLOCKED) {
      node.state = NodeState.EXPLORED;
      this.gameState.player.exploredNodes.add(nodeId);
    }
  }

  completeNode(nodeId: string): void {
    const node = this.gameState.nodes.get(nodeId);
    if (node) {
      node.state = NodeState.COMPLETED;
    }
  }

  unlockNode(nodeId: string): void {
    const node = this.gameState.nodes.get(nodeId);
    if (node && node.state === NodeState.LOCKED) {
      node.state = NodeState.UNLOCKED;
    }
  }

  addResource(resourceId: string, amount: number): void {
    const current = this.gameState.player.resources.get(resourceId) || 0;
    this.gameState.player.resources.set(resourceId, current + amount);
  }

  removeResource(resourceId: string, amount: number): boolean {
    const current = this.gameState.player.resources.get(resourceId) || 0;
    if (current >= amount) {
      this.gameState.player.resources.set(resourceId, current - amount);
      return true;
    }
    return false;
  }

  addExperience(amount: number): void {
    this.gameState.player.experience += amount;
    this.checkLevelUp();
  }

  private checkLevelUp(): void {
    const requiredExp = this.gameState.player.level * 100;
    if (this.gameState.player.experience >= requiredExp) {
      this.gameState.player.level++;
      this.gameState.player.experience -= requiredExp;
      this.gameState.player.maxHealth += 20;
      this.gameState.player.health = this.gameState.player.maxHealth;
    }
  }

  takeDamage(amount: number): void {
    this.gameState.player.health = Math.max(0, this.gameState.player.health - amount);
  }

  heal(amount: number): void {
    this.gameState.player.health = Math.min(
      this.gameState.player.maxHealth,
      this.gameState.player.health + amount
    );
  }

  setGamePhase(phase: IGameState['gamePhase']): void {
    this.gameState.gamePhase = phase;
  }

  getGamePhase(): IGameState['gamePhase'] {
    return this.gameState.gamePhase;
  }

  canAccessNode(nodeId: string): boolean {
    const node = this.gameState.nodes.get(nodeId);
    if (!node) return false;

    if (node.state === NodeState.LOCKED) {
      if (!node.requirements) return false;
      
      return node.requirements.every(req => 
        this.gameState.player.resources.has(req) && 
        this.gameState.player.resources.get(req)! > 0
      );
    }

    return true;
  }

  getConnectedNodes(nodeId: string): NodeData[] {
    const node = this.gameState.nodes.get(nodeId);
    if (!node) return [];

    return node.connections
      .map(connId => this.gameState.nodes.get(connId))
      .filter((node): node is NodeData => node !== undefined);
  }
} 