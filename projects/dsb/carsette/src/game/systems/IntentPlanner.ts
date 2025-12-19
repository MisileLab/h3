import { EnemyIntent, EnemyType, TilePos } from '../episodes/types';

export interface PlannerContext {
  enemyId: string;
  enemyType: EnemyType;
  enemyPos: TilePos;
  allies: Array<{ id: string; pos: TilePos }>;
  barriers: TilePos[];
  obstacles: TilePos[];
}

const manhattan = (a: TilePos, b: TilePos): number => Math.abs(a.x - b.x) + Math.abs(a.y - b.y);

export const planEnemyIntent = (ctx: PlannerContext): EnemyIntent => {
  const closestAlly = ctx.allies
    .slice()
    .sort((a, b) => manhattan(a.pos, ctx.enemyPos) - manhattan(b.pos, ctx.enemyPos))[0];

  const fallback: TilePos = closestAlly ? { ...closestAlly.pos } : { x: ctx.enemyPos.x, y: ctx.enemyPos.y };

  if (ctx.enemyType === 'JAMMER') {
    return {
      enemyId: ctx.enemyId,
      type: 'ATTACK_TILE',
      targetTile: fallback,
      meta: {
        damage: 1,
        heatOnHit: 1,
        range: 4,
      },
    };
  }

  // BLOCKER: alternate between barrier and burst depending on distance.
  const distance = closestAlly ? manhattan(closestAlly.pos, ctx.enemyPos) : 999;
  const wantsBarrier = distance <= 3;

  if (wantsBarrier) {
    return {
      enemyId: ctx.enemyId,
      type: 'PLACE_BARRIER',
      targetTile: fallback,
      meta: {
        barrierDurationTurns: 2,
      },
    };
  }

  return {
    enemyId: ctx.enemyId,
    type: 'ATTACK_TILE',
    targetTile: fallback,
    meta: {
      damage: 1,
      range: 5,
    },
  };
};
