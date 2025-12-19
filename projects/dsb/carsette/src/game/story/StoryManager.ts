import { EP1_OUTSKIRTS_SCRIPT } from './ep1OutskirtsScript';
import { EpisodeStoryScript, StoryLine, StoryOnceScope, StoryTrigger } from './types';

const SCRIPTS: Record<string, EpisodeStoryScript> = {
  ep1: EP1_OUTSKIRTS_SCRIPT,
};

export class StoryManager {
  private static instance: StoryManager;

  private episodeId: string = 'ep1';
  private firedThisRun: Set<string> = new Set();
  private firedThisSession: Set<string> = new Set();
  private readonly maxLines: number = 18;
  private lines: string[] = [];
  private consoleElement: HTMLElement | null;

  private constructor() {
    this.consoleElement = document.getElementById('story-console');
  }

  public static getInstance(): StoryManager {
    if (!StoryManager.instance) {
      StoryManager.instance = new StoryManager();
    }
    return StoryManager.instance;
  }

  public setEpisodeId(episodeId: string): void {
    this.episodeId = episodeId;
  }

  public resetRun(): void {
    this.firedThisRun.clear();
    this.lines = [];
    this.render();
  }

  public trigger(triggerId: string): void {
    const trigger = this.getTrigger(triggerId);
    if (!trigger) return;

    const onceScope = trigger.once;
    if (onceScope) {
      if (this.hasFired(triggerId, onceScope)) return;
      this.markFired(triggerId, onceScope);
    }

    trigger.lines.forEach(line => this.appendLine(line));
    this.render();
  }

  private getTrigger(triggerId: string): StoryTrigger | null {
    const script = SCRIPTS[this.episodeId];
    if (!script) return null;
    return script.triggers[triggerId] ?? null;
  }

  private hasFired(triggerId: string, scope: StoryOnceScope): boolean {
    return scope === 'session' ? this.firedThisSession.has(triggerId) : this.firedThisRun.has(triggerId);
  }

  private markFired(triggerId: string, scope: StoryOnceScope): void {
    if (scope === 'session') {
      this.firedThisSession.add(triggerId);
    } else {
      this.firedThisRun.add(triggerId);
    }
  }

  private appendLine(line: StoryLine): void {
    this.lines.push(`[${line.speaker}] ${line.text}`);
    if (this.lines.length > this.maxLines) {
      this.lines = this.lines.slice(-this.maxLines);
    }
  }

  private render(): void {
    if (!this.consoleElement) {
      this.consoleElement = document.getElementById('story-console');
    }
    if (!this.consoleElement) return;
    this.consoleElement.textContent = this.lines.join('\n');
  }
}
