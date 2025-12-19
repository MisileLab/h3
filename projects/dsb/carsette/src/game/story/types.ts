export type StorySpeaker = 'SYSTEM' | 'TUT' | 'NIKO' | 'IO';

export type StoryOnceScope = 'run' | 'session';

export interface StoryLine {
  speaker: StorySpeaker;
  text: string;
}

export interface StoryTrigger {
  once?: StoryOnceScope;
  lines: StoryLine[];
}

export interface EpisodeStoryScript {
  episodeId: string;
  triggers: Record<string, StoryTrigger>;
}
