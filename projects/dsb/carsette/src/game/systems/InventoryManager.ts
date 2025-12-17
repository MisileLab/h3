import { ItemData, ItemRotation, ItemShape, PlacedItem } from '../types/Item';
import { RunManager } from './RunManager';

/**
 * Grid dimensions for the main inventory
 */
const GRID_WIDTH = 10;
const GRID_HEIGHT = 4;

/**
 * Result interface for inventory operations
 */
interface InventoryOperationResult {
  success: boolean;
  message?: string;
}

/**
 * Singleton manager for inventory system
 * Handles main grid (10x4) and temporary tray storage
 */
export class InventoryManager {
  private static instance: InventoryManager;

  /**
   * Main inventory grid (10 columns x 4 rows)
   * Stores item IDs or null for empty cells
   */
  private mainGrid: (string | null)[][];

  /**
   * Tray for temporary item storage
   * Items here are lost if not moved to grid before turn ends
   */
  private tray: ItemData[];

  /**
   * Buffer slots for temporary storage (different from tray)
   */
  private buffer: ItemData[];

  private readonly baseBufferCapacity: number = 2;
  private jammerPenalty: number = 0;

  /**
   * Map of all items by their ID for quick lookup
   */
  private itemsById: Map<string, PlacedItem>;

  /**
   * Total scrap resources available
   */
  private scrapResources: number;

  private constructor() {
    this.mainGrid = this.createEmptyGrid();
    this.tray = [];
    this.buffer = [];
    this.itemsById = new Map();
    this.scrapResources = 0;
  }

  /**
   * Get the singleton instance
   */
  public static getInstance(): InventoryManager {
    if (!InventoryManager.instance) {
      InventoryManager.instance = new InventoryManager();
    }
    return InventoryManager.instance;
  }

  /**
   * Create an empty grid
   */
  private createEmptyGrid(): (string | null)[][] {
    const grid: (string | null)[][] = [];
    for (let y = 0; y < GRID_HEIGHT; y++) {
      grid[y] = [];
      for (let x = 0; x < GRID_WIDTH; x++) {
        grid[y][x] = null;
      }
    }
    return grid;
  }

  /**
   * Add an item to the tray
   */
  public addItemToTray(item: ItemData): void {
    this.tray.push(item);
  }

  /**
   * Move a tray item into the buffer (used by right-click shortcuts)
   */
  public moveTrayItemToBuffer(itemId: string): { success: boolean; overflow: boolean; lostItem?: ItemData } {
    const index = this.tray.findIndex(item => item.id === itemId);
    if (index === -1) {
      return { success: false, overflow: false };
    }

    const item = this.tray.splice(index, 1)[0];
    const result = this.addItemToBuffer(item);
    return { success: true, overflow: result.overflow, lostItem: result.lostItem };
  }

  /**
   * Add an item to the buffer, respecting capacity and overflow rules
   */
  public addItemToBuffer(item: ItemData): { overflow: boolean; lostItem?: ItemData } {
    this.buffer.push(item);

    const capacity = this.getBufferCapacity();
    if (this.buffer.length > capacity) {
      const lostItem = this.evictLowestValueFromBuffer();
      RunManager.getInstance().addHeat(1);
      return { overflow: true, lostItem };
    }

    return { overflow: false };
  }

  /**
   * Remove the lowest scrap value item from the buffer (DATA SPILL)
   */
  private evictLowestValueFromBuffer(): ItemData | undefined {
    if (this.buffer.length === 0) {
      return undefined;
    }

    let lowestIndex = 0;
    for (let i = 1; i < this.buffer.length; i++) {
      if (this.buffer[i].scrapValue < this.buffer[lowestIndex].scrapValue) {
        lowestIndex = i;
      }
    }

    const [removed] = this.buffer.splice(lowestIndex, 1);
    return removed;
  }

  /**
   * Get all items in the tray
   */
  public getTray(): ItemData[] {
    return [...this.tray];
  }

  public getBuffer(): ItemData[] {
    return [...this.buffer];
  }

  public setJammerPenalty(isActive: boolean): void {
    this.jammerPenalty = isActive ? 1 : 0;
  }

  public getBufferCapacity(): number {
    return Math.max(0, this.baseBufferCapacity - this.jammerPenalty);
  }

  /**
   * Get the main grid state
   */
  public getGrid(): (string | null)[][] {
    return this.mainGrid.map(row => [...row]);
  }

  /**
   * Get an item by its ID
   */
  public getItemById(itemId: string): ItemData | null {
    const placed = this.itemsById.get(itemId);
    return placed ? placed.item : null;
  }

  /**
   * Get current scrap resources
   */
  public getScrapResources(): number {
    return this.scrapResources;
  }

  public addScrap(amount: number): void {
    this.scrapResources = Math.max(0, this.scrapResources + amount);
  }

  /**
   * Rotate an item's shape 90 degrees clockwise
   */
  private rotateShape(shape: ItemShape): ItemShape {
    const rows = shape.length;
    const cols = shape[0].length;
    const rotated: ItemShape = [];

    for (let x = 0; x < cols; x++) {
      rotated[x] = [];
      for (let y = rows - 1; y >= 0; y--) {
        rotated[x][rows - 1 - y] = shape[y][x];
      }
    }

    return rotated;
  }

  /**
   * Get the rotated shape for an item at a given rotation
   */
  private getRotatedShape(item: ItemData, rotation: ItemRotation): ItemShape {
    let shape = item.shape;
    const rotations = rotation % 4;

    for (let i = 0; i < rotations; i++) {
      shape = this.rotateShape(shape);
    }

    return shape;
  }

  /**
   * Check if an item can be placed at the given position
   */
  private canPlaceItem(
    itemId: string,
    x: number,
    y: number,
    shape: ItemShape
  ): boolean {
    const rows = shape.length;
    const cols = shape[0].length;

    // Check bounds
    if (x < 0 || y < 0 || x + cols > GRID_WIDTH || y + rows > GRID_HEIGHT) {
      return false;
    }

    // Check for collisions with other items
    for (let dy = 0; dy < rows; dy++) {
      for (let dx = 0; dx < cols; dx++) {
        if (shape[dy][dx] === 1) {
          const cellValue = this.mainGrid[y + dy][x + dx];
          // Cell must be empty or occupied by the same item (for rotation/movement)
          if (cellValue !== null && cellValue !== itemId) {
            return false;
          }
        }
      }
    }

    return true;
  }

  /**
   * Clear an item from the grid
   */
  private clearItemFromGrid(itemId: string): void {
    for (let y = 0; y < GRID_HEIGHT; y++) {
      for (let x = 0; x < GRID_WIDTH; x++) {
        if (this.mainGrid[y][x] === itemId) {
          this.mainGrid[y][x] = null;
        }
      }
    }
  }

  /**
   * Place an item on the grid at the given position
   */
  private placeItemOnGrid(
    itemId: string,
    x: number,
    y: number,
    shape: ItemShape
  ): void {
    const rows = shape.length;
    const cols = shape[0].length;

    for (let dy = 0; dy < rows; dy++) {
      for (let dx = 0; dx < cols; dx++) {
        if (shape[dy][dx] === 1) {
          this.mainGrid[y + dy][x + dx] = itemId;
        }
      }
    }
  }

  /**
   * Move an item to the grid at the specified position and rotation
   * Item can come from tray or be moved within the grid
   */
  public moveItem(
    itemId: string,
    targetX: number,
    targetY: number,
    rotation: ItemRotation
  ): InventoryOperationResult {
    // Check if item is in tray
    const trayIndex = this.tray.findIndex(item => item.id === itemId);
    let item: ItemData | null = null;

    if (trayIndex !== -1) {
      // Item is in tray
      item = this.tray[trayIndex];
    } else {
      // Item might be in grid
      const placed = this.itemsById.get(itemId);
      if (placed) {
        item = placed.item;
      }
    }

    if (!item) {
      return { success: false, message: 'Item not found' };
    }

    // Get rotated shape
    const rotatedShape = this.getRotatedShape(item, rotation);

    // Clear item from grid if it's already placed
    if (this.itemsById.has(itemId)) {
      this.clearItemFromGrid(itemId);
    }

    // Check if placement is valid
    if (!this.canPlaceItem(itemId, targetX, targetY, rotatedShape)) {
      // Restore item to grid if it was there before
      if (this.itemsById.has(itemId)) {
        const oldPlacement = this.itemsById.get(itemId)!;
        const oldShape = this.getRotatedShape(
          oldPlacement.item,
          oldPlacement.item.rotation
        );
        this.placeItemOnGrid(
          itemId,
          oldPlacement.position.x,
          oldPlacement.position.y,
          oldShape
        );
      }
      return { success: false, message: 'Cannot place item at this position' };
    }

    // Update item rotation
    item.rotation = rotation;

    // Place item on grid
    this.placeItemOnGrid(itemId, targetX, targetY, rotatedShape);

    // Update item tracking
    this.itemsById.set(itemId, {
      item,
      position: { x: targetX, y: targetY },
    });

    // Remove from tray if it was there
    if (trayIndex !== -1) {
      this.tray.splice(trayIndex, 1);
    }

    return { success: true };
  }

  /**
   * Rotate an item in place
   */
  public rotateItem(itemId: string): InventoryOperationResult {
    const placed = this.itemsById.get(itemId);
    if (!placed) {
      return { success: false, message: 'Item not in grid' };
    }

    const newRotation = ((placed.item.rotation + 1) % 4) as ItemRotation;
    return this.moveItem(
      itemId,
      placed.position.x,
      placed.position.y,
      newRotation
    );
  }

  /**
   * Consume ammunition from an item
   */
  public consumeAmmo(itemId: string, amount: number): InventoryOperationResult {
    const item = this.getItemById(itemId);
    if (!item) {
      return { success: false, message: 'Item not found' };
    }

    if (item.currentAmmo === null || item.maxAmmo === null) {
      return { success: false, message: 'Item does not use ammunition' };
    }

    if (item.currentAmmo < amount) {
      return { success: false, message: 'Insufficient ammunition' };
    }

    item.currentAmmo -= amount;

    // Auto-scrap if ammo depleted
    if (item.currentAmmo === 0) {
      return this.scrapItem(itemId);
    }

    return { success: true };
  }

  /**
   * Scrap an item and gain resources
   */
  public scrapItem(itemId: string): InventoryOperationResult {
    const placed = this.itemsById.get(itemId);
    if (!placed) {
      return { success: false, message: 'Item not in grid' };
    }

    // Add scrap value to resources
    this.scrapResources += placed.item.scrapValue;

    // Remove from grid
    this.clearItemFromGrid(itemId);

    // Remove from tracking
    this.itemsById.delete(itemId);

    return {
      success: true,
      message: `Scrapped ${placed.item.name} for ${placed.item.scrapValue} resources`,
    };
  }

  /**
   * Clear all items from the tray
   * Called at turn end - all tray items are lost
   */
  public clearTray(): void {
    this.tray = [];
  }

  /**
   * Reset the entire inventory (for testing or new game)
   */
  public reset(): void {
    this.mainGrid = this.createEmptyGrid();
    this.tray = [];
    this.buffer = [];
    this.itemsById.clear();
    this.scrapResources = 0;
  }

  /**
   * Get all placed items
   */
  public getPlacedItems(): PlacedItem[] {
    return Array.from(this.itemsById.values());
  }
}
