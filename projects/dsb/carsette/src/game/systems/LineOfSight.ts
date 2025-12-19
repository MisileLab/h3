import { TilePos } from '../episodes/types';

// Bresenham line for tile grids, excluding origin tile.
export const getLineTiles = (from: TilePos, to: TilePos): TilePos[] => {
  const tiles: TilePos[] = [];

  let x0 = from.x;
  let y0 = from.y;
  const x1 = to.x;
  const y1 = to.y;

  const dx = Math.abs(x1 - x0);
  const dy = -Math.abs(y1 - y0);
  const sx = x0 < x1 ? 1 : -1;
  const sy = y0 < y1 ? 1 : -1;
  let err = dx + dy;

  while (true) {
    if (!(x0 === from.x && y0 === from.y)) {
      tiles.push({ x: x0, y: y0 });
    }

    if (x0 === x1 && y0 === y1) {
      break;
    }

    const e2 = 2 * err;
    if (e2 >= dy) {
      err += dy;
      x0 += sx;
    }
    if (e2 <= dx) {
      err += dx;
      y0 += sy;
    }
  }

  return tiles;
};

export const hasLineOfSight = (from: TilePos, to: TilePos, blocked: Set<string>): boolean => {
  const line = getLineTiles(from, to);
  // exclude destination tile, allow hitting through unit occupancy.
  for (let i = 0; i < line.length - 1; i++) {
    const t = line[i];
    if (blocked.has(`${t.x},${t.y}`)) return false;
  }
  return true;
};
