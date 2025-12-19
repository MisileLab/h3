import { InventoryManager } from '../game/systems/InventoryManager';
import { RunManager } from '../game/systems/RunManager';
import { StoryManager } from '../game/story/StoryManager';
import { ItemData, ItemRotation } from '../game/types/Item';
import { UIManager } from './UIManager';

/**
 * Drag state interface
 */
interface DragState {
  isDragging: boolean;
  itemId: string | null;
  offsetX: number;
  offsetY: number;
  dragElement: HTMLElement | null;
  sourceType: 'tray' | 'grid' | null;
  currentRotation: ItemRotation;
}

/**
 * Context menu state interface
 */
interface ContextMenuState {
  isOpen: boolean;
  itemId: string | null;
  menuElement: HTMLElement | null;
}

/**
 * Grid cell size in pixels
 */
const CELL_SIZE = 40;
const GRID_WIDTH = 10;
const GRID_HEIGHT = 4;

/**
 * Singleton UI manager for inventory visualization and interaction
 * Integrates with InventoryManager for data management
 */
export class InventoryUI {
  private static instance: InventoryUI;

  private inventoryManager: InventoryManager;
  private containerElement: HTMLElement | null = null;
  private trayElement: HTMLElement | null = null;
  private bufferElement: HTMLElement | null = null;
  private bufferLabelElement: HTMLElement | null = null;
  private gridElement: HTMLElement | null = null;
  private scrapValueElement: HTMLElement | null = null;

  private dragState: DragState = {
    isDragging: false,
    itemId: null,
    offsetX: 0,
    offsetY: 0,
    dragElement: null,
    sourceType: null,
    currentRotation: 0,
  };

  private contextMenu: ContextMenuState = {
    isOpen: false,
    itemId: null,
    menuElement: null,
  };

  private constructor() {
    this.inventoryManager = InventoryManager.getInstance();
    this.initialize();
  }

  public static getInstance(): InventoryUI {
    if (!InventoryUI.instance) {
      InventoryUI.instance = new InventoryUI();
    }
    return InventoryUI.instance;
  }

  /**
   * Initialize the inventory UI
   */
  private initialize(): void {
    this.createInventoryStructure();
    this.setupEventListeners();
    this.update();
  }

  /**
   * Create the HTML structure for inventory
   */
  private createInventoryStructure(): void {
    const inventoryGridElement = document.getElementById('inventory-grid');
    if (!inventoryGridElement) {
      console.error('inventory-grid element not found');
      return;
    }

    inventoryGridElement.innerHTML = '';

    // Create container
    this.containerElement = document.createElement('div');
    this.containerElement.id = 'inventory-container';
    this.containerElement.style.cssText = `
      display: flex;
      flex-direction: column;
      gap: 16px;
      padding: 8px;
      height: 100%;
    `;

    // Create scrap value display
    const scrapHeader = document.createElement('div');
    scrapHeader.style.cssText = `
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 4px 8px;
      background: rgba(89, 65, 0, 0.3);
      border: 1px solid #FFB000;
      font-family: 'VT323', monospace;
      font-size: 16px;
      color: #FFB000;
    `;
    scrapHeader.innerHTML = `
      <span>SCRAP RESOURCES</span>
      <span id="scrap-value">0</span>
    `;
    this.scrapValueElement = scrapHeader.querySelector('#scrap-value');

    // Create tray section
    const traySection = document.createElement('div');
    traySection.style.cssText = `
      display: flex;
      flex-direction: column;
      gap: 8px;
    `;

    const trayLabel = document.createElement('div');
    trayLabel.textContent = 'TRAY (Temporary Storage)';
    trayLabel.style.cssText = `
      font-family: 'VT323', monospace;
      font-size: 14px;
      color: #FFB000;
      text-transform: uppercase;
    `;

    this.trayElement = document.createElement('div');
    this.trayElement.id = 'inventory-tray';
    this.trayElement.style.cssText = `
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      min-height: 60px;
      padding: 8px;
      background: rgba(176, 38, 255, 0.1);
      border: 2px solid #B026FF;
      border-radius: 4px;
    `;

    traySection.appendChild(trayLabel);
    traySection.appendChild(this.trayElement);

    // Buffer section
    const bufferSection = document.createElement('div');
    bufferSection.style.cssText = `
      display: flex;
      flex-direction: column;
      gap: 6px;
    `;

    this.bufferLabelElement = document.createElement('div');
    this.bufferLabelElement.textContent = 'BUFFER (2)';
    this.bufferLabelElement.style.cssText = `
      font-family: 'VT323', monospace;
      font-size: 14px;
      color: #FFB000;
      text-transform: uppercase;
    `;

    this.bufferElement = document.createElement('div');
    this.bufferElement.id = 'inventory-buffer';
    this.bufferElement.style.cssText = `
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      min-height: 48px;
      padding: 6px;
      background: rgba(255, 255, 255, 0.05);
      border: 2px solid #FFB000;
      border-radius: 4px;
    `;

    bufferSection.appendChild(this.bufferLabelElement);
    bufferSection.appendChild(this.bufferElement);

    // Create main grid section
    const gridSection = document.createElement('div');
    gridSection.style.cssText = `
      display: flex;
      flex-direction: column;
      gap: 8px;
      flex: 1;
    `;

    const gridLabel = document.createElement('div');
    gridLabel.textContent = `MAIN INVENTORY (${GRID_WIDTH}×${GRID_HEIGHT})`;
    gridLabel.style.cssText = `
      font-family: 'VT323', monospace;
      font-size: 14px;
      color: #FFB000;
      text-transform: uppercase;
    `;

    const gridWrapper = document.createElement('div');
    gridWrapper.style.cssText = `
      position: relative;
      display: inline-block;
    `;

    this.gridElement = document.createElement('div');
    this.gridElement.id = 'inventory-main-grid';
    this.gridElement.style.cssText = `
      display: grid;
      grid-template-columns: repeat(${GRID_WIDTH}, ${CELL_SIZE}px);
      grid-template-rows: repeat(${GRID_HEIGHT}, ${CELL_SIZE}px);
      gap: 1px;
      background: #594100;
      padding: 1px;
      border: 2px solid #FFB000;
      position: relative;
    `;

    // Create grid cells
    for (let y = 0; y < GRID_HEIGHT; y++) {
      for (let x = 0; x < GRID_WIDTH; x++) {
        const cell = document.createElement('div');
        cell.className = 'grid-cell';
        cell.dataset.gridX = x.toString();
        cell.dataset.gridY = y.toString();
        cell.style.cssText = `
          background: rgba(255, 176, 0, 0.1);
          border: 1px solid rgba(89, 65, 0, 0.5);
          transition: background 0.1s;
        `;
        this.gridElement.appendChild(cell);
      }
    }

    gridWrapper.appendChild(this.gridElement);
    gridSection.appendChild(gridLabel);
    gridSection.appendChild(gridWrapper);

    // Assemble structure
    this.containerElement.appendChild(scrapHeader);
    this.containerElement.appendChild(traySection);
    this.containerElement.appendChild(bufferSection);
    this.containerElement.appendChild(gridSection);
    inventoryGridElement.appendChild(this.containerElement);
  }

  /**
   * Setup event listeners for drag and drop
   */
  private setupEventListeners(): void {
    // Mouse events for drag and drop
    document.addEventListener('mousedown', this.handleMouseDown.bind(this));
    document.addEventListener('mousemove', this.handleMouseMove.bind(this));
    document.addEventListener('mouseup', this.handleMouseUp.bind(this));

    // Keyboard events for rotation
    document.addEventListener('keydown', this.handleKeyDown.bind(this));

    // Context menu
    document.addEventListener('contextmenu', this.handleContextMenu.bind(this));

    // Close context menu on click outside
    document.addEventListener('click', this.handleClickOutside.bind(this));
  }

  /**
   * Handle mouse down event
   */
  private handleMouseDown(event: MouseEvent): void {
    const target = event.target as HTMLElement;

    // Check if clicking on an item
    const itemElement = target.closest('.inventory-item') as HTMLElement;
    if (!itemElement || event.button !== 0) return;

    event.preventDefault();

    const itemId = itemElement.dataset.itemId;
    if (!itemId) return;

    // Determine source type first
    const sourceType = itemElement.closest('#inventory-tray')
      ? 'tray'
      : 'grid';

    // Get item from appropriate source
    let item: ItemData | null = null;
    if (sourceType === 'tray') {
      // Find item in tray
      const trayItems = this.inventoryManager.getTray();
      item = trayItems.find(i => i.id === itemId) || null;
    } else {
      // Find item in grid
      item = this.inventoryManager.getItemById(itemId);
    }

    if (!item) {
      console.warn('Item not found:', itemId);
      return;
    }

    // Create drag ghost
    const dragGhost = itemElement.cloneNode(true) as HTMLElement;
    dragGhost.style.position = 'fixed';
    dragGhost.style.pointerEvents = 'none';
    dragGhost.style.opacity = '0.7';
    dragGhost.style.zIndex = '10000';
    dragGhost.style.transform = 'none';
    document.body.appendChild(dragGhost);

    // Calculate offset
    const rect = itemElement.getBoundingClientRect();
    const offsetX = event.clientX - rect.left;
    const offsetY = event.clientY - rect.top;

    this.dragState = {
      isDragging: true,
      itemId,
      offsetX,
      offsetY,
      dragElement: dragGhost,
      sourceType,
      currentRotation: item.rotation,
    };

    // Position drag ghost
    dragGhost.style.left = `${event.clientX - offsetX}px`;
    dragGhost.style.top = `${event.clientY - offsetY}px`;

    // Hide original
    itemElement.style.opacity = '0.3';
  }

  /**
   * Handle mouse move event
   */
  private handleMouseMove(event: MouseEvent): void {
    if (!this.dragState.isDragging || !this.dragState.dragElement) return;

    // Update drag ghost position
    this.dragState.dragElement.style.left = `${event.clientX - this.dragState.offsetX}px`;
    this.dragState.dragElement.style.top = `${event.clientY - this.dragState.offsetY}px`;

    // Highlight grid cells under cursor
    this.highlightGridCells(event.clientX, event.clientY);
  }

  /**
   * Handle mouse up event
   */
  private handleMouseUp(event: MouseEvent): void {
    if (!this.dragState.isDragging) return;

    const itemId = this.dragState.itemId;
    if (!itemId) {
      this.resetDragState();
      return;
    }

    // Check if dropped on grid
    const gridPos = this.getGridPositionFromMouse(event.clientX, event.clientY);

    if (gridPos) {
      // Attempt to place item
      const result = this.inventoryManager.moveItem(
        itemId,
        gridPos.x,
        gridPos.y,
        this.dragState.currentRotation
      );

      if (!result.success) {
        console.warn('Failed to place item:', result.message);
      }
    } else {
      // Check if dropped outside inventory (discard)
      const isOverInventory = this.isOverInventoryArea(event.clientX, event.clientY);
      
      if (!isOverInventory && this.dragState.sourceType === 'grid') {
        // Discard item from grid
        if (confirm('Discard this item?')) {
          this.inventoryManager.scrapItem(itemId);
        }
      }
    }

    this.resetDragState();
    this.update();
  }

  /**
   * Handle key down event
   */
  private handleKeyDown(event: KeyboardEvent): void {
    if (!this.dragState.isDragging) return;

    if (event.key === 'r' || event.key === 'R') {
      event.preventDefault();
      
      // Rotate item
      this.dragState.currentRotation = ((this.dragState.currentRotation + 1) % 4) as ItemRotation;
      
      // Update drag ghost rotation
      if (this.dragState.dragElement && this.dragState.itemId) {
        // Get item from appropriate source
        let item: ItemData | null = null;
        if (this.dragState.sourceType === 'tray') {
          const trayItems = this.inventoryManager.getTray();
          item = trayItems.find(i => i.id === this.dragState.itemId) || null;
        } else {
          item = this.inventoryManager.getItemById(this.dragState.itemId);
        }
        
        if (item) {
          this.updateDragGhostRotation(item);
        }
      }
    }
  }

  /**
   * Update drag ghost visual rotation
   */
  private updateDragGhostRotation(item: ItemData): void {
    if (!this.dragState.dragElement) return;

    const shape = this.getRotatedShape(item.shape, this.dragState.currentRotation);
    const width = shape[0].length * CELL_SIZE;
    const height = shape.length * CELL_SIZE;

    this.dragState.dragElement.style.width = `${width}px`;
    this.dragState.dragElement.style.height = `${height}px`;
  }

  /**
   * Handle context menu (right click)
   */
  private handleContextMenu(event: MouseEvent): void {
    const target = event.target as HTMLElement;
    const itemElement = target.closest('.inventory-item') as HTMLElement;

    if (!itemElement) return;

    event.preventDefault();

    const itemId = itemElement.dataset.itemId;
    if (!itemId) return;

    const item = this.inventoryManager.getItemById(itemId);
    if (!item) return;

    this.showContextMenu(event.clientX, event.clientY, itemId, item);
  }

  /**
   * Show context menu
   */
  private showContextMenu(x: number, y: number, itemId: string, item: ItemData): void {
    this.closeContextMenu();

    const menu = document.createElement('div');
    menu.className = 'context-menu';
    menu.style.cssText = `
      position: fixed;
      left: ${x}px;
      top: ${y}px;
      background: rgba(10, 10, 10, 0.95);
      border: 2px solid #FFB000;
      padding: 8px;
      z-index: 10001;
      font-family: 'VT323', monospace;
      font-size: 16px;
      color: #FFB000;
      min-width: 150px;
    `;

    const runManager = RunManager.getInstance();
    const uiManager = UIManager.getInstance();
    const storyManager = StoryManager.getInstance();
    storyManager.setEpisodeId(runManager.getEpisode().id);

    // Item info
    const info = document.createElement('div');
    info.style.cssText = `
      padding: 8px;
      border-bottom: 1px solid #594100;
      margin-bottom: 4px;
    `;
    info.innerHTML = `
      <div style="font-weight: bold;">${item.name}</div>
      <div style="font-size: 14px; color: #B026FF;">${item.type}</div>
      ${item.currentAmmo !== null ? `<div style="font-size: 14px;">Ammo: ${item.currentAmmo}/${item.maxAmmo}</div>` : ''}
      <div style="font-size: 14px;">Scrap: ${item.scrapValue}</div>
      ${item.name === 'COOLANT CAPSULE' ? `<div style="font-size: 12px; color: #8cf;">HEAT MITIGATIONS USED: ${runManager.getCoolantMitigationsUsed()}/2</div>` : ''}
    `;

    // Actions
    const actions = document.createElement('div');
    actions.style.cssText = `
      display: flex;
      flex-direction: column;
      gap: 4px;
    `;

    // Rotate action
    const rotateBtn = this.createMenuButton('Rotate [R]', () => {
      const result = this.inventoryManager.rotateItem(itemId);
      if (result.success) {
        this.update();
      }
      this.closeContextMenu();
    });

    const isInTray = this.inventoryManager.getTray().some(trayItem => trayItem.id === itemId);
    if (isInTray) {
      const bufferBtn = this.createMenuButton('Send to Buffer', () => {
        const result = this.inventoryManager.moveTrayItemToBuffer(itemId);
        if (result.success) {
          storyManager.trigger('TRG_BUFFER_EXPLAIN');
          if (result.overflow) {
            uiManager.updateHeat(runManager.getHeat());
            uiManager.updateSystemMessage('DATA SPILL // BUFFER OVERFLOW');
            storyManager.trigger('TRG_BUFFER_OVERFLOW_FIRST');
          }
          this.update();
        }
        this.closeContextMenu();
      });
      actions.appendChild(bufferBtn);
    }

    if (item.name === 'COOLANT CAPSULE') {
      const stabBtn = this.createMenuButton('USE: +1 STAB CHARGE', () => {
        this.inventoryManager.consumeItem(itemId);
        runManager.addStabilizerCharge(1);
        uiManager.updateStabilizer(runManager.getStabilizerCharges());
        uiManager.updateSystemMessage('COOLANT USED // STAB +1');
        this.update();
        this.closeContextMenu();
      });

      const heatBtn = this.createMenuButton('USE: HEAT -1 (CAP 2)', () => {
        if (!runManager.registerCoolantHeatMitigation()) {
          uiManager.updateSystemMessage('HEAT MITIGATION CAP REACHED');
          this.closeContextMenu();
          return;
        }
        this.inventoryManager.consumeItem(itemId);
        runManager.reduceHeat(1);
        uiManager.updateHeat(runManager.getHeat());
        uiManager.updateSystemMessage('COOLANT USED // HEAT -1');
        this.update();
        this.closeContextMenu();
      });

      actions.appendChild(stabBtn);
      actions.appendChild(heatBtn);
    }

    // Scrap action
    const scrapBtn = this.createMenuButton(`Scrap (+${item.scrapValue})`, () => {
      if (confirm(`Scrap ${item.name} for ${item.scrapValue} resources?`)) {
        this.inventoryManager.scrapItem(itemId);
        this.update();
      }
      this.closeContextMenu();
    });
    scrapBtn.style.color = '#FF0000';

    actions.appendChild(rotateBtn);
    actions.appendChild(scrapBtn);

    menu.appendChild(info);
    menu.appendChild(actions);
    document.body.appendChild(menu);

    this.contextMenu = {
      isOpen: true,
      itemId,
      menuElement: menu,
    };
  }

  /**
   * Create a menu button
   */
  private createMenuButton(label: string, onClick: () => void): HTMLElement {
    const button = document.createElement('div');
    button.textContent = label;
    button.style.cssText = `
      padding: 4px 8px;
      background: rgba(255, 176, 0, 0.1);
      border: 1px solid #FFB000;
      cursor: pointer;
      transition: background 0.1s;
    `;

    button.addEventListener('mouseenter', () => {
      button.style.background = 'rgba(255, 176, 0, 0.3)';
    });

    button.addEventListener('mouseleave', () => {
      button.style.background = 'rgba(255, 176, 0, 0.1)';
    });

    button.addEventListener('click', onClick);

    return button;
  }

  /**
   * Close context menu
   */
  private closeContextMenu(): void {
    if (this.contextMenu.menuElement) {
      this.contextMenu.menuElement.remove();
      this.contextMenu = {
        isOpen: false,
        itemId: null,
        menuElement: null,
      };
    }
  }

  /**
   * Handle click outside to close context menu
   */
  private handleClickOutside(event: MouseEvent): void {
    if (!this.contextMenu.isOpen) return;

    const target = event.target as HTMLElement;
    if (!target.closest('.context-menu')) {
      this.closeContextMenu();
    }
  }

  /**
   * Highlight grid cells under cursor during drag
   */
  private highlightGridCells(mouseX: number, mouseY: number): void {
    if (!this.gridElement) return;

    // Clear previous highlights
    const cells = this.gridElement.querySelectorAll('.grid-cell');
    cells.forEach(cell => {
      (cell as HTMLElement).style.background = 'rgba(255, 176, 0, 0.1)';
    });

    const gridPos = this.getGridPositionFromMouse(mouseX, mouseY);
    if (!gridPos || !this.dragState.itemId) return;

    // Get item from appropriate source
    let item: ItemData | null = null;
    if (this.dragState.sourceType === 'tray') {
      const trayItems = this.inventoryManager.getTray();
      item = trayItems.find(i => i.id === this.dragState.itemId) || null;
    } else {
      item = this.inventoryManager.getItemById(this.dragState.itemId);
    }
    
    if (!item) return;

    const shape = this.getRotatedShape(item.shape, this.dragState.currentRotation);
    const canPlace = this.canPlaceAt(gridPos.x, gridPos.y, shape, this.dragState.itemId);

    // Highlight cells
    for (let dy = 0; dy < shape.length; dy++) {
      for (let dx = 0; dx < shape[0].length; dx++) {
        if (shape[dy][dx] === 1) {
          const cellX = gridPos.x + dx;
          const cellY = gridPos.y + dy;

          if (cellX >= 0 && cellX < GRID_WIDTH && cellY >= 0 && cellY < GRID_HEIGHT) {
            const cell = this.getCellElement(cellX, cellY);
            if (cell) {
              cell.style.background = canPlace
                ? 'rgba(0, 255, 0, 0.3)'
                : 'rgba(255, 0, 0, 0.3)';
            }
          }
        }
      }
    }
  }

  /**
   * Get grid cell element
   */
  private getCellElement(x: number, y: number): HTMLElement | null {
    if (!this.gridElement) return null;
    const index = y * GRID_WIDTH + x;
    return this.gridElement.children[index] as HTMLElement;
  }

  /**
   * Get grid position from mouse coordinates
   */
  private getGridPositionFromMouse(mouseX: number, mouseY: number): { x: number; y: number } | null {
    if (!this.gridElement) return null;

    const rect = this.gridElement.getBoundingClientRect();
    const relX = mouseX - rect.left;
    const relY = mouseY - rect.top;

    if (relX < 0 || relY < 0 || relX > rect.width || relY > rect.height) {
      return null;
    }

    const x = Math.floor(relX / (CELL_SIZE + 1));
    const y = Math.floor(relY / (CELL_SIZE + 1));

    if (x < 0 || x >= GRID_WIDTH || y < 0 || y >= GRID_HEIGHT) {
      return null;
    }

    return { x, y };
  }

  /**
   * Check if mouse is over inventory area
   */
  private isOverInventoryArea(mouseX: number, mouseY: number): boolean {
    if (!this.containerElement) return false;
    const rect = this.containerElement.getBoundingClientRect();
    return (
      mouseX >= rect.left &&
      mouseX <= rect.right &&
      mouseY >= rect.top &&
      mouseY <= rect.bottom
    );
  }

  /**
   * Check if item can be placed at position
   */
  private canPlaceAt(x: number, y: number, shape: number[][], itemId: string): boolean {
    const grid = this.inventoryManager.getGrid();

    for (let dy = 0; dy < shape.length; dy++) {
      for (let dx = 0; dx < shape[0].length; dx++) {
        if (shape[dy][dx] === 1) {
          const cellX = x + dx;
          const cellY = y + dy;

          if (cellX < 0 || cellX >= GRID_WIDTH || cellY < 0 || cellY >= GRID_HEIGHT) {
            return false;
          }

          const cellValue = grid[cellY][cellX];
          if (cellValue !== null && cellValue !== itemId) {
            return false;
          }
        }
      }
    }

    return true;
  }

  /**
   * Get rotated shape
   */
  private getRotatedShape(shape: number[][], rotation: ItemRotation): number[][] {
    let result = shape;
    for (let i = 0; i < rotation; i++) {
      result = this.rotateShapeOnce(result);
    }
    return result;
  }

  /**
   * Rotate shape 90 degrees clockwise
   */
  private rotateShapeOnce(shape: number[][]): number[][] {
    const rows = shape.length;
    const cols = shape[0].length;
    const rotated: number[][] = [];

    for (let x = 0; x < cols; x++) {
      rotated[x] = [];
      for (let y = rows - 1; y >= 0; y--) {
        rotated[x][rows - 1 - y] = shape[y][x];
      }
    }

    return rotated;
  }

  /**
   * Reset drag state
   */
  private resetDragState(): void {
    if (this.dragState.dragElement) {
      this.dragState.dragElement.remove();
    }

    // Restore original item opacity
    if (this.dragState.itemId) {
      const items = document.querySelectorAll(`[data-item-id="${this.dragState.itemId}"]`);
      items.forEach(item => {
        (item as HTMLElement).style.opacity = '1';
      });
    }

    // Clear grid highlights
    if (this.gridElement) {
      const cells = this.gridElement.querySelectorAll('.grid-cell');
      cells.forEach(cell => {
        (cell as HTMLElement).style.background = 'rgba(255, 176, 0, 0.1)';
      });
    }

    this.dragState = {
      isDragging: false,
      itemId: null,
      offsetX: 0,
      offsetY: 0,
      dragElement: null,
      sourceType: null,
      currentRotation: 0,
    };
  }

  /**
   * Update the entire UI from InventoryManager state
   */
  public update(): void {
    this.updateScrapValue();
    this.updateTray();
    this.updateBuffer();
    this.updateGrid();
  }

  /**
   * Update scrap value display
   */
  private updateScrapValue(): void {
    if (!this.scrapValueElement) return;
    this.scrapValueElement.textContent = this.inventoryManager.getScrapResources().toString();
  }

  /**
   * Update tray display
   */
  private updateTray(): void {
    if (!this.trayElement) return;

    this.trayElement.innerHTML = '';
    const trayItems = this.inventoryManager.getTray();

    if (trayItems.length === 0) {
      const emptyMessage = document.createElement('div');
      emptyMessage.textContent = 'Empty';
      emptyMessage.style.cssText = `
        color: rgba(255, 176, 0, 0.5);
        font-family: 'VT323', monospace;
        font-size: 14px;
        padding: 8px;
      `;
      this.trayElement.appendChild(emptyMessage);
      return;
    }

    trayItems.forEach(item => {
      const itemElement = this.createItemElement(item);
      this.trayElement!.appendChild(itemElement);
    });
  }

  private updateBuffer(): void {
    if (!this.bufferElement || !this.bufferLabelElement) return;

    const bufferElement = this.bufferElement;
    bufferElement.innerHTML = '';
    const bufferItems = this.inventoryManager.getBuffer();
    this.bufferLabelElement.textContent = `BUFFER (${this.inventoryManager.getBufferCapacity()})`;

    if (bufferItems.length === 0) {
      const emptyMessage = document.createElement('div');
      emptyMessage.textContent = 'Empty';
      emptyMessage.style.cssText = `
        color: rgba(255, 176, 0, 0.5);
        font-family: 'VT323', monospace;
        font-size: 14px;
        padding: 4px;
      `;
      bufferElement.appendChild(emptyMessage);
      return;
    }

    bufferItems.forEach(item => {
      const itemElement = this.createItemElement(item);
      itemElement.style.background = 'rgba(89, 65, 0, 0.4)';
      itemElement.style.border = '2px solid #FFB000';
      bufferElement.appendChild(itemElement);
    });
  }

  /**
   * Update grid display
   */
  private updateGrid(): void {
    if (!this.gridElement) return;

    // Remove existing items
    const existingItems = this.gridElement.querySelectorAll('.inventory-item');
    existingItems.forEach(item => item.remove());

    // Place items
    const placedItems = this.inventoryManager.getPlacedItems();
    placedItems.forEach(({ item, position }) => {
      const itemElement = this.createItemElement(item);
      const shape = this.getRotatedShape(item.shape, item.rotation);
      
      itemElement.style.position = 'absolute';
      itemElement.style.left = `${position.x * (CELL_SIZE + 1) + 1}px`;
      itemElement.style.top = `${position.y * (CELL_SIZE + 1) + 1}px`;
      itemElement.style.width = `${shape[0].length * (CELL_SIZE + 1) - 1}px`;
      itemElement.style.height = `${shape.length * (CELL_SIZE + 1) - 1}px`;

      this.gridElement!.appendChild(itemElement);
    });
  }

  /**
   * Create an item DOM element
   */
  private createItemElement(item: ItemData): HTMLElement {
    const shape = this.getRotatedShape(item.shape, item.rotation);
    const width = shape[0].length * CELL_SIZE;
    const height = shape.length * CELL_SIZE;

    const element = document.createElement('div');
    element.className = 'inventory-item';
    element.dataset.itemId = item.id;
    element.style.cssText = `
      width: ${width}px;
      height: ${height}px;
      background: rgba(176, 38, 255, 0.5);
      border: 2px solid #B026FF;
      border-radius: 4px;
      cursor: grab;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 4px;
      box-sizing: border-box;
      transition: transform 0.1s;
      font-family: 'VT323', monospace;
      color: #FFFFFF;
      text-align: center;
      overflow: hidden;
    `;

    // Item name
    const nameElement = document.createElement('div');
    nameElement.textContent = item.name;
    nameElement.style.cssText = `
      font-size: 12px;
      font-weight: bold;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      width: 100%;
    `;

    // Ammo display (if applicable)
    if (item.currentAmmo !== null && item.maxAmmo !== null) {
      const ammoElement = document.createElement('div');
      ammoElement.textContent = `${item.currentAmmo}/${item.maxAmmo}`;
      ammoElement.style.cssText = `
        font-size: 10px;
        color: #FFB000;
      `;
      element.appendChild(ammoElement);
    }

    element.appendChild(nameElement);

    // Hover effect
    element.addEventListener('mouseenter', () => {
      if (!this.dragState.isDragging) {
        element.style.transform = 'scale(1.05)';
        element.style.zIndex = '100';
      }
    });

    element.addEventListener('mouseleave', () => {
      if (!this.dragState.isDragging) {
        element.style.transform = 'scale(1)';
        element.style.zIndex = 'auto';
      }
    });

    return element;
  }

  /**
   * Add a test item to tray (for development/testing)
   */
  public addTestItem(): void {
    const testItem: ItemData = {
      id: `item-${Date.now()}`,
      name: 'Test Item',
      type: 'WEAPON' as any,
      shape: [
        [1, 1],
        [1, 0],
      ],
      rotation: 0,
      maxAmmo: 30,
      currentAmmo: 30,
      scrapValue: 10,
    };

    this.inventoryManager.addItemToTray(testItem);
    this.update();
  }
}
