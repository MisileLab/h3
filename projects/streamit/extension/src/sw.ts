import type {
  ExtensionSettings,
  PopupToSwMessage,
  ServerMessage,
  AudioFrameMessage,
  ContentScriptMessage,
} from "./types";

let offscreenPort: chrome.runtime.Port | null = null;
let offscreenAudioPort: chrome.runtime.Port | null = null;
let ws: WebSocket | null = null;
let wsUrl: string | null = null;
let storedUserToken: string | null = null;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let reconnectAttempts = 0;
let captureTabId: number | null = null;
let startInProgress = false;
const MAX_RECONNECT_DELAY = 10000;

async function getActiveVideoKey(): Promise<string | null> {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id || !tab.url) {
    return null;
  }
  try {
    const url = new URL(tab.url);
    if (url.hostname.includes("youtube.com")) {
      const videoId = url.searchParams.get("v");
      return videoId ? `youtube:${videoId}` : null;
    }
    if (url.hostname.includes("twitch.tv")) {
      const channel = url.pathname.split("/").filter(Boolean)[0];
      return channel ? `twitch:${channel}` : null;
    }
  } catch {
    return null;
  }
  return null;
}

async function createOffscreenDocument(): Promise<void> {
  if (await chrome.offscreen.hasDocument()) {
    return;
  }

  await chrome.offscreen.createDocument({
    url: "offscreen.html",
    reasons: [chrome.offscreen.Reason.AUDIO_PLAYBACK],
    justification: "Need AudioWorklet for audio processing",
  });
}

async function sendToOffscreen(message: object): Promise<void> {
  if (!offscreenPort) {
    await createOffscreenDocument();
    offscreenPort = chrome.runtime.connect({ name: "sw" });
    offscreenPort.onDisconnect.addListener(() => {
      offscreenPort = null;
    });
  }

  try {
    offscreenPort.postMessage(message);
  } catch (error) {
    console.warn("Failed to post to offscreen:", error);
    offscreenPort = null;
    await createOffscreenDocument();
    offscreenPort = chrome.runtime.connect({ name: "sw" });
    offscreenPort.onDisconnect.addListener(() => {
      offscreenPort = null;
    });
    offscreenPort.postMessage(message);
  }
}

async function handleStartCaptions(message: {
  serverUrl: string;
  userToken: string;
  language: string;
  targetLanguage: string;
  tabId: number;
  settings: ExtensionSettings;
}): Promise<void> {
  if (startInProgress) {
    return;
  }
  startInProgress = true;

  const { serverUrl, userToken, language, targetLanguage, tabId, settings } =
    message;

  wsUrl = serverUrl;
  storedUserToken = userToken;

  try {
    const videoKey = await getActiveVideoKey();
    if (videoKey) {
      const stored = await chrome.storage.sync.get(`disabled:${videoKey}`);
      if (stored[`disabled:${videoKey}`]) {
        sendStatusUpdate("disconnected", "Captions disabled for this video");
        return;
      }
    }

    const capturedTabs = await chrome.tabCapture.getCapturedTabs();
    const activeCapture = capturedTabs.find((capture) => capture.tabId === tabId);
    const hasActiveCapture = Boolean(
      activeCapture &&
        activeCapture.status !== "stopped" &&
        activeCapture.status !== "error",
    );
    if (hasActiveCapture) {
      sendStatusUpdate("connected", "Already capturing this tab");
      return;
    }

    if (ws && ws.readyState === WebSocket.OPEN) {
      sendStatusUpdate("connected", "Already connected");
      return;
    }

    if (ws && ws.readyState === WebSocket.CONNECTING) {
      sendStatusUpdate("reconnecting", "Connecting...");
      return;
    }

    if (ws && ws.readyState === WebSocket.CLOSING) {
      await cleanup();
    }

    // Verify tab exists
    await chrome.tabs.get(tabId);

    const streamId = await chrome.tabCapture.getMediaStreamId({
      targetTabId: tabId,
    });

    if (!streamId) {
      throw new Error("Failed to get tab audio stream");
    }

    await sendToOffscreen({
      type: "START_AUDIO",
      streamId,
    });

    await connectWebSocket(language, targetLanguage, settings);
    captureTabId = tabId;
    sendStatusUpdate("connected", "Connected");
  } catch (error) {
    console.error("Failed to start captions:", error);
    const errorMessage =
      error instanceof Error ? error.message : "Unknown error";
    sendStatusUpdate("error", `Start failed: ${errorMessage}`);
    await cleanup();
  } finally {
    startInProgress = false;
  }
}

async function connectWebSocket(
  language: string,
  targetLanguage: string,
  settings: ExtensionSettings,
): Promise<boolean> {
  const target = targetLanguage;

  function scheduleReconnect(
    reconnectLanguage: string,
    reconnectSettings: ExtensionSettings,
  ): void {
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
    }

    reconnectAttempts++;
    const delay = Math.min(
      1000 * Math.pow(2, reconnectAttempts),
      MAX_RECONNECT_DELAY,
    );

    sendStatusUpdate("reconnecting", `Reconnecting in ${delay / 1000}s...`);

    reconnectTimer = setTimeout(async () => {
      await connectWebSocket(reconnectLanguage, target, reconnectSettings);
    }, delay);
  }

  if (!wsUrl || !storedUserToken) {
    sendStatusUpdate("error", "Missing config");
    return false;
  }

  try {
    const url = new URL(wsUrl);
    url.searchParams.set("token", storedUserToken);

    ws = new WebSocket(url.toString());
    reconnectAttempts = 0;

    ws.binaryType = "arraybuffer";

    ws.onopen = async () => {
      console.log("WebSocket connected");
      reconnectAttempts = 0;

      const startMessage = {
        type: "start",
        lang: language,
        targetLang: targetLanguage,
        clientSessionId: generateUUID(),
        platformHint: await detectPlatform(),
      };

      ws?.send(JSON.stringify(startMessage));

      await sendToOffscreen({
        type: "UPDATE_SETTINGS",
        settings,
      });
    };

    ws.onmessage = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data as string) as ServerMessage;
        handleServerMessage(data);
      } catch (error) {
        console.error("Failed to parse server message:", error);
      }
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      sendStatusUpdate("error", "Connection error");
    };

    ws.onclose = () => {
      console.log("WebSocket closed");
      sendStatusUpdate("disconnected", "Disconnected");

      if (reconnectAttempts < 10) {
        scheduleReconnect(language, settings);
      }
    };

    return true;
  } catch (error) {
    console.error("WebSocket connection failed:", error);
    const errorMessage =
      error instanceof Error ? error.message : "Unknown error";
    sendStatusUpdate("error", `Connection failed: ${errorMessage}`);
    return false;
  }
}


function handleServerMessage(data: ServerMessage): void {
  if (data.type === "partial") {
    sendToActiveTab({
      type: "CAPTION_PARTIAL",
      text: data.text,
    });
  } else if (data.type === "final") {
    sendToActiveTab({
      type: "CAPTION_FINAL",
      text: data.text,
    });
  } else if (data.type === "info") {
    console.log("Info from server:", data.message);
    if (typeof data.secondsUsed === "number") {
      sendStatusUpdate(
        "connected",
        `Connected (${data.secondsUsed.toFixed(1)}s)`,
      );
    } else if (data.message) {
      sendStatusUpdate("connected", data.message);
    }
  } else if (data.type === "error") {
    console.error("Error from server:", data.message);
    sendStatusUpdate("error", data.message);
    cleanup();
  }
}

async function handleStopCaptions(): Promise<void> {
  console.log("Stopping captions");
  sendToActiveTab({ type: "CLEAR_CAPTIONS" });
  await cleanup();
  sendStatusUpdate("disconnected", "Stopped");
}

function sendToActiveTab(message: ContentScriptMessage): void {
  if (captureTabId === null) {
    return;
  }

  chrome.tabs.sendMessage(captureTabId, message, () => {
    if (chrome.runtime.lastError) {
      return;
    }
  });
}

async function handleUpdateSettings(message: {
  settings: ExtensionSettings;
}): Promise<void> {
  await sendToOffscreen({
    type: "UPDATE_SETTINGS",
    settings: message.settings,
  });

  if (ws && ws.readyState === WebSocket.OPEN) {
    await sendToOffscreen({
      type: "RESUME_AUDIO",
    });
  }

  if (message.settings.captionsMuted) {
    sendToActiveTab({ type: "CLEAR_CAPTIONS" });
  }
}

function sendStatusUpdate(status: string, message: string): void {
  chrome.runtime.sendMessage({
    type: "STATUS_UPDATE",
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

  await sendToOffscreen({ type: "STOP_AUDIO" });

  if (ws) {
    ws.close();
    ws = null;
  }

  if (offscreenAudioPort) {
    offscreenAudioPort.disconnect();
    offscreenAudioPort = null;
  }

  wsUrl = null;
  storedUserToken = null;
  captureTabId = null;
}

function generateUUID(): string {
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

async function detectPlatform(): Promise<string> {
  try {
    const [tab] = await chrome.tabs.query({
      active: true,
      currentWindow: true,
    });
    if (tab?.url) {
      const { hostname } = new URL(tab.url);
      if (hostname.includes("youtube")) {
        return "youtube";
      }
      if (hostname.includes("twitch")) {
        return "twitch";
      }
    }
  } catch {
    // Ignore errors
  }
  return "unknown";
}

chrome.runtime.onConnect.addListener((port) => {
  if (port.name === "offscreen") {
    offscreenAudioPort = port;

    port.onMessage.addListener(async (message: AudioFrameMessage) => {
      if (
        message.type === "AUDIO_FRAME" &&
        ws &&
        ws.readyState === WebSocket.OPEN
      ) {
        ws.send(message.data);
      }
    });

    port.onDisconnect.addListener(() => {
      offscreenAudioPort = null;
    });
  }
});

chrome.runtime.onMessage.addListener((message: PopupToSwMessage, _sender, sendResponse) => {
  if (message.type === "GET_VIDEO_KEY") {
    const tabId = message.tabId;
    chrome.tabs.get(tabId, (tab) => {
      if (!tab?.url) {
        sendResponse(null);
        return;
      }
      try {
        const url = new URL(tab.url);
        if (url.hostname.includes("youtube.com")) {
          const videoId = url.searchParams.get("v");
          sendResponse(videoId ? `youtube:${videoId}` : null);
          return;
        }
        if (url.hostname.includes("twitch.tv")) {
          const channel = url.pathname.split("/").filter(Boolean)[0];
          sendResponse(channel ? `twitch:${channel}` : null);
          return;
        }
      } catch {
        sendResponse(null);
        return;
      }
      sendResponse(null);
    });
    return true;
  }

  switch (message.type) {
    case "START_CAPTIONS":
      handleStartCaptions(message);
      break;
    case "STOP_CAPTIONS":
      handleStopCaptions();
      break;
    case "UPDATE_SETTINGS":
      handleUpdateSettings(message);
      break;
  }
  return false;
});
