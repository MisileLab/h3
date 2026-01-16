// Shared types for StreamIt extension

export interface CaptionSettings {
  fontSize: number;
  backgroundOpacity: number;
  position: 'top' | 'bottom';
  maxLines: number;
}

export interface ExtensionSettings extends CaptionSettings {
  serverUrl: string;
  userToken: string;
  language: string;
}

// Messages from popup/background to content script
export interface CaptionPartialMessage {
  type: 'CAPTION_PARTIAL';
  text: string;
}

export interface CaptionFinalMessage {
  type: 'CAPTION_FINAL';
  text: string;
}

export interface UpdateSettingsMessage {
  type: 'UPDATE_SETTINGS';
  settings: CaptionSettings;
}

export interface ClearCaptionsMessage {
  type: 'CLEAR_CAPTIONS';
}

export type ContentScriptMessage =
  | CaptionPartialMessage
  | CaptionFinalMessage
  | UpdateSettingsMessage
  | ClearCaptionsMessage;

// Messages from popup to service worker
export interface StartCaptionsMessage {
  type: 'START_CAPTIONS';
  serverUrl: string;
  userToken: string;
  language: string;
  tabId: number;
  settings: ExtensionSettings;
}

export interface StopCaptionsMessage {
  type: 'STOP_CAPTIONS';
}

export interface UpdateSettingsSwMessage {
  type: 'UPDATE_SETTINGS';
  settings: ExtensionSettings;
}

export type PopupToSwMessage =
  | StartCaptionsMessage
  | StopCaptionsMessage
  | UpdateSettingsSwMessage;

// Messages from service worker to popup
export interface StatusUpdateMessage {
  type: 'STATUS_UPDATE';
  status: 'connected' | 'disconnected' | 'error' | 'reconnecting';
  message: string;
}

// Messages for offscreen communication
export interface StartAudioMessage {
  type: 'START_AUDIO';
  streamId: string;
}

export interface StopAudioMessage {
  type: 'STOP_AUDIO';
}

export interface ResumeAudioMessage {
  type: 'RESUME_AUDIO';
}

export interface OffscreenUpdateSettingsMessage {
  type: 'UPDATE_SETTINGS';
  settings: CaptionSettings;
}

export type SwToOffscreenMessage =
  | StartAudioMessage
  | StopAudioMessage
  | ResumeAudioMessage
  | OffscreenUpdateSettingsMessage;

// Messages from offscreen to service worker
export interface AudioFrameMessage {
  type: 'AUDIO_FRAME';
  data: ArrayBuffer;
}

// Server protocol messages
export interface ServerStartMessage {
  type: 'start';
  lang: string;
  clientSessionId: string;
  platformHint: string;
}

export interface ServerPartialMessage {
  type: 'partial';
  text: string;
}

export interface ServerFinalMessage {
  type: 'final';
  text: string;
}

export interface ServerInfoMessage {
  type: 'info';
  message: string;
  secondsUsed?: number;
}

export interface ServerErrorMessage {
  type: 'error';
  message: string;
}

export type ServerMessage =
  | ServerPartialMessage
  | ServerFinalMessage
  | ServerInfoMessage
  | ServerErrorMessage;

// Audio worklet messages
export interface WorkletInitMessage {
  type: 'INIT';
  sampleRate?: number;
  frameSize?: number;
}

export interface WorkletAudioFrameMessage {
  type: 'AUDIO_FRAME';
  data: ArrayBuffer;
}

export type WorkletMessage = WorkletInitMessage | WorkletAudioFrameMessage;
