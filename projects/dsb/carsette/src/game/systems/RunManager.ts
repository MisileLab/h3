import { episode1 } from '../data/episode1';
import { EnemyIntent, EpisodeConfig, NodeConfig } from '../types/Run';

interface RunProgress {
  currentNodeIndex: number;
  powerRestored: boolean;
  productionComplete: boolean;
  extractionComplete: boolean;
}

export class RunManager {
  private static instance: RunManager;

  private readonly episode: EpisodeConfig = episode1;
  private progress: RunProgress = {
    currentNodeIndex: 0,
    powerRestored: false,
    productionComplete: false,
    extractionComplete: false,
  };
  private heat: number = 0;
  private stabilizerCharges: number = 1;
  private pendingIntents: EnemyIntent[] = [];
  private extractionStages: Record<string, number> = {};

  public static getInstance(): RunManager {
    if (!RunManager.instance) {
      RunManager.instance = new RunManager();
    }
    return RunManager.instance;
  }

  public reset(): void {
    this.progress = {
      currentNodeIndex: 0,
      powerRestored: false,
      productionComplete: false,
      extractionComplete: false,
    };
    this.heat = 0;
    this.stabilizerCharges = 1;
    this.pendingIntents = [];
    this.extractionStages = {};
  }

  public getEpisode(): EpisodeConfig {
    return this.episode;
  }

  public getCurrentNode(): NodeConfig {
    return this.episode.nodes[this.progress.currentNodeIndex];
  }

  public advanceNode(): void {
    if (this.progress.currentNodeIndex < this.episode.nodes.length - 1) {
      this.progress.currentNodeIndex += 1;
    }
  }

  public setCurrentNodeById(nodeId: string): void {
    const index = this.episode.nodes.findIndex(node => node.id === nodeId);
    if (index !== -1) {
      this.progress.currentNodeIndex = index;
    }
  }

  public markPowerRestored(): void {
    this.progress.powerRestored = true;
  }

  public markProductionComplete(): void {
    this.progress.productionComplete = true;
  }

  public markExtractionComplete(): void {
    this.progress.extractionComplete = true;
  }

  public isRunComplete(): boolean {
    return (
      this.progress.powerRestored &&
      this.progress.productionComplete &&
      this.progress.extractionComplete
    );
  }

  public getHeat(): number {
    return this.heat;
  }

  public addHeat(amount: number): void {
    this.heat = Math.min(10, this.heat + amount);
  }

  public getStabilizerCharges(): number {
    return this.stabilizerCharges;
  }

  public consumeStabilizer(): boolean {
    if (this.stabilizerCharges <= 0) {
      return false;
    }
    this.stabilizerCharges -= 1;
    this.addHeat(1);
    return true;
  }

  public setPendingIntents(intents: EnemyIntent[]): void {
    this.pendingIntents = intents;
  }

  public getPendingIntents(): EnemyIntent[] {
    return this.pendingIntents;
  }

  public clearPendingIntents(): void {
    this.pendingIntents = [];
  }

  public getExtractionStage(nodeId: string): number {
    return this.extractionStages[nodeId] ?? 0;
  }

  public incrementExtractionStage(nodeId: string): number {
    const next = (this.extractionStages[nodeId] ?? 0) + 1;
    this.extractionStages[nodeId] = next;
    return next;
  }
}
