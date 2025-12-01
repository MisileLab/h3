// Types for the Meeting Assistant application

export interface SubtitleEntry {
  id: string;
  timestamp: number;
  speaker: string | null;
  text_en: string;
  text_ko: string;
}

export interface ActionItem {
  text: string;
  isRelevantToUser: boolean;
}

export interface MeetingSummary {
  summary: string[];
  decisions: string[];
  action_items: ActionItem[];
}

export interface SessionStatus {
  is_active: boolean;
  start_time: number | null;
}

export interface UserSettings {
  username: string;
  keywords: string[];
  apiKey: string;
}

export interface SubtitleEvent {
  subtitle: SubtitleEntry;
}

export interface SummaryEvent {
  summary: MeetingSummary;
}

export interface ErrorEvent {
  message: string;
}
