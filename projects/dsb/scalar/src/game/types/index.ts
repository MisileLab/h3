export enum NodeType {
  START = 'start',
  EXPLORATION = 'exploration',
  RESOURCE = 'resource',
  ENEMY = 'enemy',
  TREASURE = 'treasure',
  BOSS = 'boss',
  EXIT = 'exit'
}

export enum NodeState {
  LOCKED = 'locked',
  UNLOCKED = 'unlocked',
  EXPLORED = 'explored',
  COMPLETED = 'completed'
}

export interface NodeData {
  id: string;
  x: number;
  y: number;
  type: NodeType;
  state: NodeState;
  connections: string[];
  name: string;
  description: string;
  resources?: ResourceData[];
  enemies?: EnemyData[];
  requirements?: string[];
}

export interface ResourceData {
  id: string;
  name: string;
  amount: number;
  maxAmount: number;
  rarity: 'common' | 'uncommon' | 'rare' | 'epic' | 'legendary';
}

export interface EnemyData {
  id: string;
  name: string;
  health: number;
  maxHealth: number;
  attack: number;
  defense: number;
  level: number;
}

export interface PlayerData {
  level: number;
  experience: number;
  health: number;
  maxHealth: number;
  resources: Map<string, number>;
  exploredNodes: Set<string>;
  currentNode: string;
  inventory: InventoryItem[];
}

export interface InventoryItem {
  id: string;
  name: string;
  type: 'weapon' | 'armor' | 'consumable' | 'material';
  rarity: 'common' | 'uncommon' | 'rare' | 'epic' | 'legendary';
  quantity: number;
  maxQuantity: number;
  stats?: Record<string, number>;
}

export interface GameState {
  player: PlayerData;
  nodes: Map<string, NodeData>;
  currentNode: string;
  gamePhase: 'menu' | 'exploration' | 'combat' | 'inventory';
} 