export type EpisodeId = 'ep1' | 'ep2' | 'ep3';

export interface StoryProgressStateV1 {
  version: 1;
  completed: EpisodeId[];
}

const STORAGE_KEY = 'carsette.storyProgress.v1';
const EPISODE_ORDER: EpisodeId[] = ['ep1', 'ep2', 'ep3'];

const DEFAULT_STATE: StoryProgressStateV1 = {
  version: 1,
  completed: [],
};

const isEpisodeId = (value: unknown): value is EpisodeId => {
  return value === 'ep1' || value === 'ep2' || value === 'ep3';
};

const sortEpisodes = (episodes: EpisodeId[]): EpisodeId[] => {
  return [...episodes].sort((a, b) => EPISODE_ORDER.indexOf(a) - EPISODE_ORDER.indexOf(b));
};

const normalizeState = (value: unknown): StoryProgressStateV1 => {
  if (!value || typeof value !== 'object') return { ...DEFAULT_STATE };

  const version = (value as { version?: unknown }).version;
  if (version !== 1) return { ...DEFAULT_STATE };

  const completedRaw = (value as { completed?: unknown }).completed;
  if (!Array.isArray(completedRaw)) return { ...DEFAULT_STATE };

  const completed: EpisodeId[] = [];
  completedRaw.forEach(item => {
    if (isEpisodeId(item) && !completed.includes(item)) completed.push(item);
  });

  return {
    version: 1,
    completed: sortEpisodes(completed),
  };
};

export class ProgressManager {
  private static instance: ProgressManager;

  private state: StoryProgressStateV1;

  private constructor() {
    this.state = this.load();
  }

  public static getInstance(): ProgressManager {
    if (!ProgressManager.instance) {
      ProgressManager.instance = new ProgressManager();
    }
    return ProgressManager.instance;
  }

  public getCompletedEpisodes(): EpisodeId[] {
    return [...this.state.completed];
  }

  public getNextEpisodeId(): EpisodeId {
    const completed = new Set(this.state.completed);
    const next = EPISODE_ORDER.find(id => !completed.has(id));
    return next ?? 'ep3';
  }

  public markEpisodeComplete(episodeId: string): void {
    if (!isEpisodeId(episodeId)) return;

    const expected = this.getNextEpisodeId();
    if (episodeId !== expected) return;

    if (this.state.completed.includes(episodeId)) return;

    this.state = {
      version: 1,
      completed: sortEpisodes([...this.state.completed, episodeId]),
    };
    this.save(this.state);
  }

  public resetProgress(): void {
    this.state = { ...DEFAULT_STATE };
    this.save(this.state);
  }

  private load(): StoryProgressStateV1 {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return { ...DEFAULT_STATE };
      return normalizeState(JSON.parse(raw));
    } catch {
      return { ...DEFAULT_STATE };
    }
  }

  private save(state: StoryProgressStateV1): void {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch {
      // Ignore persistence failures.
    }
  }
}
