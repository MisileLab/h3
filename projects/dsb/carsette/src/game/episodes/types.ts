export interface TilePos {
  x: number;
  y: number;
}

export type EpisodeId = 'EP1_OUTSKIRTS';

export type NodeId = 'N0' | 'N1' | 'N2' | 'N3A' | 'N3B' | 'RESULT';

export type NodeType = 'COMBAT' | 'BLUEPRINT' | 'RESULT';

export interface EpisodeConfig {
  id: EpisodeId;
  name: string;
  startNodeId: NodeId;
  nodes: Record<NodeId, NodeConfig>;
}

export interface NodeConfig {
  id: NodeId;
  name: string;
  type: NodeType;
  next: NodeId[];
  encounter?: EncounterConfig;
  blueprint?: BlueprintConfig;
}

export type EnemyType = 'BLOCKER' | 'JAMMER';

export interface EnemySpawnConfig {
  id: string;
  type: EnemyType;
  pos: TilePos;
  weakened?: boolean;
}

export type InteractableType = 'CHEST_SCRAP' | 'BREAKER' | 'EXTRACT_CONSOLE';

export interface InteractableConfig {
  id: string;
  type: InteractableType;
  pos: TilePos;
  scrapAmount?: number;
}

export type EncounterMode = 'GATEHOUSE' | 'SUBSTATION' | 'EXTRACT_A' | 'EXTRACT_B';

export interface EncounterConfig {
  id: string;
  mode: EncounterMode;
  width: number;
  height: number;
  obstacles: TilePos[];
  alliesSpawn: TilePos[];
  enemies: EnemySpawnConfig[];
  interactables: InteractableConfig[];
}

export interface BlueprintConfig {
  id: string;
  scrapCostPerMachine: number;
  productionGoal: {
    basicAttackCassette: number;
    patchKit: number;
  };
}

export type EnemyIntentType = 'ATTACK_TILE' | 'PLACE_BARRIER';

export interface EnemyIntent {
  enemyId: string;
  type: EnemyIntentType;
  targetTile: TilePos;
  meta: {
    damage?: number;
    heal?: number;
    heatOnHit?: number;
    barrierDurationTurns?: number;
    range?: number;
  };
}
