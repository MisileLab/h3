import { create } from 'zustand';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import type { 
  SubtitleEntry, 
  MeetingSummary, 
  SessionStatus, 
  UserSettings,
  SubtitleEvent,
  SummaryEvent,
  ErrorEvent
} from '../types';

interface MeetingStore {
  // Session state
  isSessionActive: boolean;
  sessionStartTime: number | null;
  
  // Subtitles
  subtitles: SubtitleEntry[];
  currentSubtitle: SubtitleEntry | null;
  
  // Summary
  summary: MeetingSummary;
  
  // Settings
  settings: UserSettings;
  isSettingsOpen: boolean;
  
  // Error state
  error: string | null;
  
  // Loading state
  isLoading: boolean;

  // View state
  showEndSummary: boolean;
  finalSummary: MeetingSummary | null;
  
  // Actions
  setApiKey: (apiKey: string) => Promise<void>;
  setKeywords: (keywords: string[]) => Promise<void>;
  updateSettings: (settings: Partial<UserSettings>) => void;
  startSession: () => Promise<void>;
  stopSession: () => Promise<void>;
  toggleSettings: () => void;
  clearError: () => void;
  closeSummaryView: () => void;
  addSubtitle: (subtitle: SubtitleEntry) => void;
  updateSummary: (summary: MeetingSummary) => void;
  initializeListeners: () => Promise<() => void>;
  checkKeywordMatch: (text: string) => boolean;
}

const defaultSummary: MeetingSummary = {
  summary: [],
  decisions: [],
  action_items: [],
};

const defaultSettings: UserSettings = {
  username: '',
  keywords: [],
  apiKey: '',
};

export const useMeetingStore = create<MeetingStore>((set, get) => ({
  // Initial state
  isSessionActive: false,
  sessionStartTime: null,
  subtitles: [],
  currentSubtitle: null,
  summary: defaultSummary,
  settings: defaultSettings,
  isSettingsOpen: false,
  error: null,
  isLoading: false,
  showEndSummary: false,
  finalSummary: null,

  // Actions
  setApiKey: async (apiKey: string) => {
    try {
      await invoke('set_api_key', { apiKey });
      set((state) => ({
        settings: { ...state.settings, apiKey },
        error: null,
      }));
    } catch (error) {
      set({ error: String(error) });
      throw error;
    }
  },

  setKeywords: async (keywords: string[]) => {
    try {
      await invoke('set_user_keywords', { keywords });
      set((state) => ({
        settings: { ...state.settings, keywords },
      }));
    } catch (error) {
      set({ error: String(error) });
      throw error;
    }
  },

  updateSettings: (newSettings: Partial<UserSettings>) => {
    set((state) => ({
      settings: { ...state.settings, ...newSettings },
    }));
  },

  startSession: async () => {
    set({ isLoading: true, error: null });
    try {
      await invoke('start_session');
      set({
        isSessionActive: true,
        sessionStartTime: Date.now(),
        subtitles: [],
        currentSubtitle: null,
        summary: defaultSummary,
        isLoading: false,
        showEndSummary: false,
        finalSummary: null,
      });
    } catch (error) {
      set({ error: String(error), isLoading: false });
      throw error;
    }
  },

  stopSession: async () => {
    set({ isLoading: true });
    try {
      const finalSummary = await invoke<MeetingSummary>('stop_session');
      set({
        isSessionActive: false,
        isLoading: false,
        showEndSummary: true,
        finalSummary,
      });
    } catch (error) {
      set({ error: String(error), isLoading: false });
      throw error;
    }
  },

  toggleSettings: () => {
    set((state) => ({ isSettingsOpen: !state.isSettingsOpen }));
  },

  clearError: () => {
    set({ error: null });
  },

  closeSummaryView: () => {
    set({ showEndSummary: false, finalSummary: null });
  },

  addSubtitle: (subtitle: SubtitleEntry) => {
    set((state) => ({
      subtitles: [...state.subtitles, subtitle],
      currentSubtitle: subtitle,
    }));
  },

  updateSummary: (summary: MeetingSummary) => {
    set({ summary });
  },

  checkKeywordMatch: (text: string) => {
    const { keywords } = get().settings;
    if (keywords.length === 0) return false;
    const lowerText = text.toLowerCase();
    return keywords.some((keyword) => 
      lowerText.includes(keyword.toLowerCase())
    );
  },

  initializeListeners: async () => {
    const unlistenSubtitle = await listen<SubtitleEvent>('subtitle', (event) => {
      get().addSubtitle(event.payload.subtitle);
    });

    const unlistenSummary = await listen<SummaryEvent>('summary', (event) => {
      get().updateSummary(event.payload.summary);
    });

    const unlistenError = await listen<ErrorEvent>('error', (event) => {
      set({ error: event.payload.message });
    });

    const unlistenStatus = await listen<SessionStatus>('session-status', (event) => {
      set({
        isSessionActive: event.payload.is_active,
        sessionStartTime: event.payload.start_time,
      });
    });

    // Return cleanup function
    return () => {
      unlistenSubtitle();
      unlistenSummary();
      unlistenError();
      unlistenStatus();
    };
  },
}));
