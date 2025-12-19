import { InventoryManager } from '../systems/InventoryManager';
import { EpisodeConfig, NodeId } from '../episodes/types';
import { EP1_OUTSKIRTS } from '../episodes/ep1Outskirts';

export type ExtractionChoice = 'A' | 'B' | null;

export interface RunGoalsState {
  powerRestored: boolean;
  producedBasicAttackCassette: boolean;
  producedPatchKit: boolean;
  extractionChoice: ExtractionChoice;
}

export interface BlueprintState {
  extractorBuilt: boolean;
  refinerBuilt: boolean;
  pressBuilt: boolean;
  plates: number;
}

export interface RunState {
  episodeId: string;
  currentNodeId: NodeId;
  heat: number;
  stabilizerCharges: number;
  goals: RunGoalsState;
  blueprint: BlueprintState;
  tutorial: Record<string, boolean>;
  logs: string[];
}

export class RunManager {
  private static instance: RunManager;

  private state: RunState;
  private episode: EpisodeConfig;

  private constructor() {
    this.episode = EP1_OUTSKIRTS;
    this.state = this.createFreshState(this.episode.startNodeId);
  }

  public static getInstance(): RunManager {
    if (!RunManager.instance) {
      RunManager.instance = new RunManager();
    }
    return RunManager.instance;
  }

  public getEpisode(): EpisodeConfig {
    return this.episode;
  }

  public getState(): RunState {
    return {
      ...this.state,
      goals: { ...this.state.goals },
      blueprint: { ...this.state.blueprint },
      tutorial: { ...this.state.tutorial },
      logs: [...this.state.logs],
    };
  }

  public resetEp1(): void {
    this.episode = EP1_OUTSKIRTS;
    this.state = this.createFreshState(this.episode.startNodeId);

    const inventory = InventoryManager.getInstance();
    inventory.reset();

    this.appendLog('RUN START // EP1: OUTSKIRTS');
    this.appendLog("NIKO: LINK STABLE. I\'LL WALK YOU THROUGH.");
    this.appendLog('SYSTEM: DON\'T PANIC. ORGANIZE.');
    this.appendLog('STABILIZER ISSUED: 1 CHARGE');

    this.tutorialOnce('controls', [
      'HELP: TAB INVENTORY  |  R ROTATE  |  B SEND->BUFFER',
      'HELP: F STABILIZER (+10s, HEAT+1)',
      'HELP: END TURN TO COMMIT ACTIONS',
    ]);
  }

  public setCurrentNode(nodeId: NodeId): void {
    this.state.currentNodeId = nodeId;
  }

  public getBlueprintState(): BlueprintState {
    return { ...this.state.blueprint };
  }

  public setBlueprintState(next: BlueprintState): void {
    this.state.blueprint = { ...next };
  }

  public appendLog(message: string): void {
    this.state.logs = [...this.state.logs, message].slice(-40);
  }

  public tutorialOnce(key: string, lines: string | string[]): void {
    if (this.state.tutorial[key]) return;
    this.state.tutorial[key] = true;

    const list = Array.isArray(lines) ? lines : [lines];
    list.forEach(line => this.appendLog(line));
  }

  public setPowerRestored(): void {
    if (!this.state.goals.powerRestored) {
      this.state.goals.powerRestored = true;
      this.appendLog('OBJECTIVE COMPLETE: POWER RESTORED');
    }
  }

  public setProducedBasicAttackCassette(): void {
    if (!this.state.goals.producedBasicAttackCassette) {
      this.state.goals.producedBasicAttackCassette = true;
      this.appendLog('PRODUCTION COMPLETE: BASIC ATTACK CASSETTE');
    }
  }

  public setProducedPatchKit(): void {
    if (!this.state.goals.producedPatchKit) {
      this.state.goals.producedPatchKit = true;
      this.appendLog('PRODUCTION COMPLETE: PATCH KIT');
    }
  }

  public setExtractionChoice(choice: ExtractionChoice): void {
    this.state.goals.extractionChoice = choice;
  }

  public addHeat(amount: number, reason: string): void {
    const before = this.state.heat;
    const nextHeat = Math.max(0, Math.min(10, this.state.heat + amount));
    const delta = nextHeat - this.state.heat;
    this.state.heat = nextHeat;

    if (delta !== 0) {
      this.appendLog(`HEAT +${delta} // ${reason}`);
    }

    if (before < 4 && nextHeat >= 4) {
      this.tutorialOnce('heat4', [
        'NIKO: HEAT LEVEL 4. EXTRACTION WILL REQUIRE +1 UPLOAD STAGE.',
        'SYSTEM: KEEP IT COOL OR PAY IN TIME.',
      ]);
    }

    if (before < 5 && nextHeat >= 5) {
      this.tutorialOnce('heat5', [
        'NIKO: HEAT LEVEL 5. RELAY TOWER MAY CALL IN A BLOCKER.',
        'SYSTEM: CONTINGENCY ACTIVE (EXTRACT B).',
      ]);
    }
  }

  public canUseStabilizer(): boolean {
    return this.state.stabilizerCharges > 0;
  }

  public consumeStabilizer(): boolean {
    if (!this.canUseStabilizer()) return false;
    this.state.stabilizerCharges -= 1;
    this.addHeat(1, 'STABILIZER');
    this.appendLog('STABILIZER USED: +10s');
    return true;
  }

  public isRunComplete(): boolean {
    const goals = this.state.goals;
    return (
      goals.powerRestored &&
      goals.producedBasicAttackCassette &&
      goals.producedPatchKit &&
      goals.extractionChoice !== null
    );
  }

  private createFreshState(startNodeId: NodeId): RunState {
    return {
      episodeId: this.episode.id,
      currentNodeId: startNodeId,
      heat: 0,
      stabilizerCharges: 1,
      goals: {
        powerRestored: false,
        producedBasicAttackCassette: false,
        producedPatchKit: false,
        extractionChoice: null,
      },
      blueprint: {
        extractorBuilt: false,
        refinerBuilt: false,
        pressBuilt: false,
        plates: 0,
      },
      tutorial: {},
      logs: [],
    };
  }
}
