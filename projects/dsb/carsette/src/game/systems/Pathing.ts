import { TilePos } from '../episodes/types';

export interface ReachableTile {
  pos: TilePos;
  dist: number;
  prev: TilePos | null;
}

const keyOf = (p: TilePos): string => `${p.x},${p.y}`;

const neighbors4 = (p: TilePos): TilePos[] => [
  { x: p.x + 1, y: p.y },
  { x: p.x - 1, y: p.y },
  { x: p.x, y: p.y + 1 },
  { x: p.x, y: p.y - 1 },
];

export const computeReachable = (
  start: TilePos,
  maxDist: number,
  width: number,
  height: number,
  blocked: Set<string>
): Map<string, ReachableTile> => {
  const result = new Map<string, ReachableTile>();
  const queue: ReachableTile[] = [{ pos: start, dist: 0, prev: null }];
  result.set(keyOf(start), queue[0]);

  while (queue.length > 0) {
    const current = queue.shift();
    if (!current) break;
    if (current.dist >= maxDist) continue;

    for (const n of neighbors4(current.pos)) {
      if (n.x < 0 || n.y < 0 || n.x >= width || n.y >= height) continue;
      const k = keyOf(n);
      if (blocked.has(k)) continue;
      if (result.has(k)) continue;

      const next: ReachableTile = {
        pos: n,
        dist: current.dist + 1,
        prev: current.pos,
      };
      result.set(k, next);
      queue.push(next);
    }
  }

  return result;
};

export const reconstructPath = (reachable: Map<string, ReachableTile>, to: TilePos): TilePos[] => {
  const path: TilePos[] = [];
  let cursor: ReachableTile | undefined = reachable.get(keyOf(to));

  while (cursor && cursor.prev) {
    path.push(cursor.pos);
    cursor = reachable.get(keyOf(cursor.prev));
  }

  return path.reverse();
};
