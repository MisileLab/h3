import { RunManager } from '../run/RunManager';

export class HeatSystem {
  private runManager: RunManager;

  public constructor(runManager: RunManager) {
    this.runManager = runManager;
  }

  public addHeat(amount: number, reason: string): void {
    this.runManager.addHeat(amount, reason);
  }
}
