/**
 * Item types available in the game
 */
export enum ItemType {
  WEAPON = 'WEAPON',
  CONSUMABLE = 'CONSUMABLE',
  ARMOR = 'ARMOR',
  ACCESSORY = 'ACCESSORY',
}

/**
 * Item rotation values (90-degree increments)
 * 0 = 0°, 1 = 90°, 2 = 180°, 3 = 270°
 */
export type ItemRotation = 0 | 1 | 2 | 3;

/**
 * Item shape represented as a 2D grid
 * 1 = occupied cell, 0 = empty cell
 * Example: [[1, 1], [1, 0]] represents an L-shape
 */
export type ItemShape = number[][];

/**
 * Core item data interface
 */
export interface ItemData {
  /**
   * Unique identifier for the item instance
   */
  id: string;

  /**
   * Display name of the item
   */
  name: string;

  /**
   * Type/category of the item
   */
  type: ItemType;

  /**
   * Shape of the item in inventory grid
   * 2D array where 1 = occupied, 0 = empty
   */
  shape: ItemShape;

  /**
   * Current rotation of the item (0-3)
   */
  rotation: ItemRotation;

  /**
   * Maximum ammunition capacity (null if not applicable)
   */
  maxAmmo: number | null;

  /**
   * Current ammunition count (null if not applicable)
   */
  currentAmmo: number | null;

  /**
   * Resource value when scrapping this item
   */
  scrapValue: number;

  /**
   * Optional description text
   */
  description?: string;

  /**
   * Optional sprite/icon key for rendering
   */
  spriteKey?: string;
}

/**
 * Position in the inventory grid
 */
export interface GridPosition {
  x: number;
  y: number;
}

/**
 * Item placement in the grid with its position
 */
export interface PlacedItem {
  item: ItemData;
  position: GridPosition;
}
