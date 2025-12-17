import { episode1 } from '../data/episode1';
import { episode2 } from '../data/episode2';
import { EnemyIntent, EpisodeConfig, NodeConfig } from '../types/Run';

interface RunProgress {
  currentNodeIndex: number;
  powerRestored: boolean;
  productionComplete: boolean;
  extractionComplete: boolean;
  rescueJoined: boolean;
  relaySecured: boolean;
  coolantCrafted: boolean;
}

export class RunManager {
  private static instance: RunManager;

  private availableEpisodes: Record<string, EpisodeConfig> = {
    [episode1.id]: episode1,
    [episode2.id]: episode2,
  };

  private episode: EpisodeConfig = episode2;
  private progress: RunProgress = {
    currentNodeIndex: 0,
    powerRestored: false,
    productionComplete: false,
    extractionComplete: false,
    rescueJoined: false,
    relaySecured: false,
    coolantCrafted: false,
  };
  private heat: number = 0;
  private stabilizerCharges: number = 1;
  private pendingIntents: EnemyIntent[] = [];
  private extractionStages: Record<string, number> = {};
  private toneFlag: string | null = null;
  private coolantHeatMitigations: number = 0;

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
      rescueJoined: false,
      relaySecured: false,
      coolantCrafted: false,
    };
    this.heat = 0;
    this.stabilizerCharges = 1;
    this.pendingIntents = [];
    this.extractionStages = {};
    this.toneFlag = null;
    this.coolantHeatMitigations = 0;
  }

  public getEpisode(): EpisodeConfig {
    return this.episode;
  }

  public setEpisode(episodeId: string): void {
    const selected = this.availableEpisodes[episodeId];
    if (selected) {
      this.episode = selected;
      this.reset();
    }
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

  public markRescueJoined(): void {
    this.progress.rescueJoined = true;
  }

  public hasRescueJoined(): boolean {
    return this.progress.rescueJoined;
  }

  public markRelaySecured(): void {
    this.progress.relaySecured = true;
  }

  public markCoolantCrafted(): void {
    this.progress.coolantCrafted = true;
  }

  public isRunComplete(): boolean {
    if (this.episode.id === 'ep2') {
      return (
        this.progress.rescueJoined &&
        this.progress.relaySecured &&
        this.progress.coolantCrafted &&
        this.progress.extractionComplete
      );
    }

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

  public reduceHeat(amount: number): void {
    this.heat = Math.max(0, this.heat - amount);
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

  public setToneFlag(flag: string): void {
    this.toneFlag = flag;
  }

  public getToneFlag(): string | null {
    return this.toneFlag;
  }

  public addStabilizerCharge(amount: number): void {
    this.stabilizerCharges += amount;
  }

  public registerCoolantHeatMitigation(): boolean {
    if (this.coolantHeatMitigations >= 2) {
      return false;
    }
    this.coolantHeatMitigations += 1;
    return true;
  }

  public getCoolantMitigationsUsed(): number {
    return this.coolantHeatMitigations;
  }
}
