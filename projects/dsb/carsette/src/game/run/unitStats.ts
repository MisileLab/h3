import { EnemyType } from '../episodes/types';

export interface UnitStats {
  name: string;
  maxHp: number;
  apPerTurn: number;
  move: number;
}

export interface AbilityStats {
  range: number;
  amount: number;
}

export const ALLY_STATS = {
  IO_DRONE: {
    name: 'I.O. DRONE',
    maxHp: 8,
    apPerTurn: 2,
    move: 5,
  },
  NIKO_ROVER: {
    name: 'NIKO ROVER',
    maxHp: 6,
    apPerTurn: 2,
    move: 4,
  },
} as const satisfies Record<string, UnitStats>;

export const ALLY_ABILITIES = {
  PULSE_SHOT: {
    range: 4,
    amount: 2,
  },
  PATCH_BEAM: {
    range: 3,
    amount: 2,
  },
} as const satisfies Record<string, AbilityStats>;

export const ENEMY_STATS: Record<EnemyType, UnitStats> = {
  BLOCKER: {
    name: 'BLOCKER',
    maxHp: 6,
    apPerTurn: 2,
    move: 3,
  },
  JAMMER: {
    name: 'JAMMER',
    maxHp: 4,
    apPerTurn: 2,
    move: 4,
  },
};
