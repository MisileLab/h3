import { InventoryUI } from './InventoryUI';

export class UIManager {
  private static instance: UIManager;
  
  private powerElement: HTMLElement | null;
  private dataElement: HTMLElement | null;
  private systemMessageElement: HTMLElement | null;
  private timerElement: HTMLElement | null;
  private dataTray: HTMLElement | null;
  private closeTrayButton: HTMLElement | null;
  private inventoryUI: InventoryUI | null = null;

  private constructor() {
    this.powerElement = document.getElementById('power-value');
    this.dataElement = document.getElementById('data-value');
    this.systemMessageElement = document.getElementById('system-message');
    this.timerElement = document.getElementById('timer-value');
    this.dataTray = document.getElementById('data-tray');
    this.closeTrayButton = document.getElementById('btn-close-tray');

    this.setupEventListeners();
    this.initializeInventoryUI();
  }

  public static getInstance(): UIManager {
    if (!UIManager.instance) {
      UIManager.instance = new UIManager();
    }
    return UIManager.instance;
  }

  private setupEventListeners(): void {
    if (this.closeTrayButton) {
      this.closeTrayButton.addEventListener('click', () => {
        this.closeInventory();
      });
    }
  }

  /**
   * Initialize the InventoryUI system
   */
  private initializeInventoryUI(): void {
    this.inventoryUI = InventoryUI.getInstance();
  }

  public updatePower(value: number): void {
    if (this.powerElement) {
      this.powerElement.textContent = value.toString();
    }
  }

  public updateData(value: number): void {
    if (this.dataElement) {
      this.dataElement.textContent = value.toString();
    }
  }

  public updateSystemMessage(message: string): void {
    if (this.systemMessageElement) {
      this.systemMessageElement.textContent = message;
    }
  }

  public updateTimer(seconds: number): void {
    if (this.timerElement) {
      this.timerElement.textContent = seconds.toString();
    }
  }

  public setTimerWarning(isWarning: boolean): void {
    if (this.timerElement) {
      if (isWarning) {
        this.timerElement.classList.add('timer-warning');
      } else {
        this.timerElement.classList.remove('timer-warning');
      }
    }
  }

  public openInventory(): void {
    if (this.dataTray) {
      this.dataTray.classList.remove('tray-hidden');
      this.dataTray.classList.add('tray-visible');
      console.log('Inventory opened');
      
      // Update inventory UI when opened
      if (this.inventoryUI) {
        this.inventoryUI.update();
      }
    }
  }

  public closeInventory(): void {
    if (this.dataTray) {
      this.dataTray.classList.remove('tray-visible');
      this.dataTray.classList.add('tray-hidden');
      console.log('Inventory closed');
    }
  }

  public toggleInventory(): void {
    if (this.dataTray) {
      if (this.dataTray.classList.contains('tray-visible')) {
        this.closeInventory();
      } else {
        this.openInventory();
      }
    }
  }

  public showGlitchEffect(duration: number = 500): void {
    const container = document.getElementById('game-container');
    if (container) {
      container.classList.add('glitch-active');
      setTimeout(() => {
        container.classList.remove('glitch-active');
      }, duration);
    }
  }

  /**
   * Get the InventoryUI instance
   */
  public getInventoryUI(): InventoryUI | null {
    return this.inventoryUI;
  }

  /**
   * @deprecated Use InventoryUI.update() instead
   */
  public populateInventoryGrid(_items: string[] = []): void {
    console.warn('populateInventoryGrid is deprecated. Use InventoryUI.update() instead.');
    if (this.inventoryUI) {
      this.inventoryUI.update();
    }
  }
}
