import type {
  ExtensionSettings,
  PopupToSwMessage,
  ServerMessage,
  AudioFrameMessage,
} from './types';

let offscreenPort: chrome.runtime.Port | null = null;
let ws: WebSocket | null = null;
let wsUrl: string | null = null;
let storedUserToken: string | null = null;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let reconnectAttempts = 0;
const MAX_RECONNECT_DELAY = 10000;

async function createOffscreenDocument(): Promise<void> {
  if (await chrome.offscreen.hasDocument()) {
    return;
  }

  await chrome.offscreen.createDocument({
    url: 'offscreen.html',
    reasons: [chrome.offscreen.Reason.AUDIO_PLAYBACK],
    justification: 'Need AudioWorklet for audio processing',
  });
}

async function sendToOffscreen(message: object): Promise<void> {
  if (!offscreenPort) {
    await createOffscreenDocument();
    offscreenPort = chrome.runtime.connect({ name: 'sw' });
  }

  offscreenPort.postMessage(message);
}

async function handleStartCaptions(message: {
  serverUrl: string;
  userToken: string;
  language: string;
  tabId: number;
  settings: ExtensionSettings;
}): Promise<void> {
  const { serverUrl, userToken, language, tabId, settings } = message;

  wsUrl = serverUrl;
  storedUserToken = userToken;

  try {
    // Verify tab exists
    await chrome.tabs.get(tabId);

    const streamId = await chrome.tabCapture.capture({audio: true, video: false}, (stream) => {
      if (stream) {
        sendToOffscreen({
          type: 'START_AUDIO',
          streamId: stream.id,
        });
      }
    });

    await sendToOffscreen({
      type: 'START_AUDIO',
      streamId,
    });

    await connectWebSocket(language, settings);
    sendStatusUpdate('connected', 'Connected');
  } catch (error) {
    console.error('Failed to start captions:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    sendStatusUpdate('error', `Start failed: ${errorMessage}`);
    await cleanup();
  }
}

async function connectWebSocket(language: string, settings: ExtensionSettings): Promise<boolean> {
  if (!wsUrl || !storedUserToken) {
    sendStatusUpdate('error', 'Missing config');
    return false;
  }

  try {
    const url = new URL(wsUrl);
    url.searchParams.set('token', storedUserToken);

    ws = new WebSocket(url.toString());
    reconnectAttempts = 0;

    ws.binaryType = 'arraybuffer';

    ws.onopen = async () => {
      console.log('WebSocket connected');
      reconnectAttempts = 0;

      const startMessage = {
        type: 'start',
        lang: language,
        clientSessionId: generateUUID(),
        platformHint: await detectPlatform(),
      };

      ws?.send(JSON.stringify(startMessage));

      await sendToOffscreen({
        type: 'UPDATE_SETTINGS',
        settings,
      });
    };

    ws.onmessage = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data as string) as ServerMessage;
        handleServerMessage(data);
      } catch (error) {
        console.error('Failed to parse server message:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      sendStatusUpdate('error', 'Connection error');
    };

    ws.onclose = () => {
      console.log('WebSocket closed');
      sendStatusUpdate('disconnected', 'Disconnected');

      if (reconnectAttempts < 10) {
        scheduleReconnect(language, settings);
      }
    };

    return true;
  } catch (error) {
    console.error('WebSocket connection failed:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    sendStatusUpdate('error', `Connection failed: ${errorMessage}`);
    return false;
  }
}

function scheduleReconnect(language: string, settings: ExtensionSettings): void {
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
  }

  reconnectAttempts++;
  const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), MAX_RECONNECT_DELAY);

  sendStatusUpdate('reconnecting', `Reconnecting in ${delay / 1000}s...`);

  reconnectTimer = setTimeout(async () => {
    await connectWebSocket(language, settings);
  }, delay);
}

function handleServerMessage(data: ServerMessage): void {
  if (data.type === 'partial') {
    chrome.tabs.query({ active: true, currentWindow: true }, ([tab]) => {
      if (tab?.id) {
        chrome.tabs.sendMessage(tab.id, {
          type: 'CAPTION_PARTIAL',
          text: data.text,
        });
      }
    });
  } else if (data.type === 'final') {
    chrome.tabs.query({ active: true, currentWindow: true }, ([tab]) => {
      if (tab?.id) {
        chrome.tabs.sendMessage(tab.id, {
          type: 'CAPTION_FINAL',
          text: data.text,
        });
      }
    });
  } else if (data.type === 'info') {
    console.log('Info from server:', data.message);
    if (data.secondsUsed !== undefined) {
      sendStatusUpdate('connected', `Connected (${data.secondsUsed.toFixed(1)}s)`);
    }
  } else if (data.type === 'error') {
    console.error('Error from server:', data.message);
    sendStatusUpdate('error', data.message);
    cleanup();
  }
}

async function handleStopCaptions(): Promise<void> {
  console.log('Stopping captions');
  await cleanup();
  sendStatusUpdate('disconnected', 'Stopped');
}

async function handleUpdateSettings(message: { settings: ExtensionSettings }): Promise<void> {
  await sendToOffscreen({
    type: 'UPDATE_SETTINGS',
    settings: message.settings,
  });

  if (ws && ws.readyState === WebSocket.OPEN) {
    await sendToOffscreen({
      type: 'RESUME_AUDIO',
    });
  }
}

function sendStatusUpdate(status: string, message: string): void {
  chrome.runtime.sendMessage({
    type: 'STATUS_UPDATE',
    status,
    message,
  });
}

async function cleanup(): Promise<void> {
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }

  reconnectAttempts = 0;

  await sendToOffscreen({ type: 'STOP_AUDIO' });

  if (ws) {
    ws.close();
    ws = null;
  }

  wsUrl = null;
  storedUserToken = null;
}

function generateUUID(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

async function detectPlatform(): Promise<string> {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab?.url) {
      const {hostname} = new URL(tab.url);
      if (hostname.includes('youtube')) {
        return 'youtube';
      }
      if (hostname.includes('twitch')) {
        return 'twitch';
      }
    }
  } catch {
    // Ignore errors
  }
  return 'unknown';
}

chrome.runtime.onConnect.addListener((port) => {
  if (port.name === 'offscreen') {
    offscreenPort = port;

    port.onMessage.addListener(async (message: AudioFrameMessage) => {
      if (message.type === 'AUDIO_FRAME' && (ws && ws.readyState === WebSocket.OPEN)) {
        ws.send(message.data);
      }
    });

    port.onDisconnect.addListener(() => {
      offscreenPort = null;
    });
  }
});

chrome.runtime.onMessage.addListener((message: PopupToSwMessage) => {
  switch (message.type) {
    case 'START_CAPTIONS':
      handleStartCaptions(message);
      break;
    case 'STOP_CAPTIONS':
      handleStopCaptions();
      break;
    case 'UPDATE_SETTINGS':
      handleUpdateSettings(message);
      break;
  }
});
