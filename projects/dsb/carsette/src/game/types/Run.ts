export interface IntentTarget {
  x: number;
  y: number;
}

export type EnemyIntentType = 'ATTACK_TILE' | 'PLACE_BARRIER' | 'POUNCE' | 'ANCHOR_PULSE' | 'IDLE';

export interface EnemyIntent {
  enemyId: string;
  type: EnemyIntentType;
  target: IntentTarget;
  meta?: Record<string, unknown>;
}

export interface UnitStats {
  id: string;
  name: string;
  maxHp: number;
  move: number;
  range: number;
  damage: number;
  heal?: number;
  ap: number;
}

export interface EnemyStats extends UnitStats {
  kind: 'BLOCKER' | 'JAMMER' | 'HUNTER' | 'WARDEN';
}

export interface EncounterConfig {
  enemies: EnemyStats[];
  loot?: string[];
  extraction?: {
    console: IntentTarget;
    zoneRadius: number;
    baseStages: number;
    branch: 'A' | 'B';
  };
  resonanceNodes?: IntentTarget[];
  anchorConsole?: IntentTarget;
  anchorStagesRequired?: number;
}

export type NodeType = 'combat' | 'blueprint' | 'result' | 'dialogue';

export interface NodeConfig {
  id: string;
  name: string;
  description: string;
  type: NodeType;
  encounter?: EncounterConfig;
  extractionHeatSpike?: number;
}

export interface EpisodeConfig {
  id: string;
  name: string;
  nodes: NodeConfig[];
}
