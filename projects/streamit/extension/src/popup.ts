import type { ExtensionSettings, StatusUpdateMessage } from "./types";

let isConnected = false;

const defaultSettings: ExtensionSettings = {
  serverUrl: "ws://localhost:8000/ws/viewer",
  userToken: "",
  language: "auto",
  targetLanguage: "auto",
  fontSize: 24,
  backgroundOpacity: 70,
  position: "bottom",
  maxLines: 2,
  captionsMuted: false,
};

function getElement<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) {
    throw new Error(`Element with id "${id}" not found`);
  }
  return el as T;
}

async function loadSettings(): Promise<void> {
  const keys = Object.keys(defaultSettings) as (keyof ExtensionSettings)[];
  const settings = await chrome.storage.sync.get(keys);
  Object.assign(defaultSettings, settings);

  getElement<HTMLInputElement>("server-url").value = defaultSettings.serverUrl;
  getElement<HTMLInputElement>("user-token").value = defaultSettings.userToken;
  getElement<HTMLSelectElement>("language").value = defaultSettings.language;
  getElement<HTMLSelectElement>("target-language").value =
    defaultSettings.targetLanguage;
  getElement<HTMLInputElement>("font-size").value = String(
    defaultSettings.fontSize,
  );

  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tab?.id) {
    const key = await chrome.runtime.sendMessage({
      type: "GET_VIDEO_KEY",
      tabId: tab.id,
    });
    if (typeof key === "string") {
      const storageKey = `disabled:${key}`;
      const stored = await chrome.storage.sync.get(storageKey);
      getElement<HTMLInputElement>("disable-current-video").checked = Boolean(
        stored[storageKey],
      );
    }
  }

  getElement<HTMLInputElement>("background-opacity").value = String(
    defaultSettings.backgroundOpacity,
  );
  getElement<HTMLSelectElement>("position").value = defaultSettings.position;
  getElement<HTMLSelectElement>("max-lines").value = String(
    defaultSettings.maxLines,
  );
  getElement<HTMLInputElement>("mute-captions").checked =
    defaultSettings.captionsMuted;

  updateDisplayValues();
}

async function saveSettings(): Promise<void> {
  const settings: ExtensionSettings = {
    serverUrl: getElement<HTMLInputElement>("server-url").value.trim(),
    userToken: getElement<HTMLInputElement>("user-token").value.trim(),
    language: getElement<HTMLSelectElement>("language").value,
    targetLanguage: getElement<HTMLSelectElement>("target-language").value,
    fontSize: parseInt(getElement<HTMLInputElement>("font-size").value),
    backgroundOpacity: parseInt(
      getElement<HTMLInputElement>("background-opacity").value,
    ),
    position: getElement<HTMLSelectElement>("position").value as
      | "top"
      | "bottom",
    maxLines: parseInt(getElement<HTMLSelectElement>("max-lines").value),
    captionsMuted: getElement<HTMLInputElement>("mute-captions").checked,
  };

  defaultSettings.targetLanguage = settings.targetLanguage;
  await chrome.storage.sync.set(settings);
  Object.assign(defaultSettings, settings);
}

function updateDisplayValues(): void {
  getElement("font-size-value").textContent = String(defaultSettings.fontSize);
  getElement("background-opacity-value").textContent = String(
    defaultSettings.backgroundOpacity,
  );
}

function setStatus(type: string, message?: string): void {
  const statusEl = getElement("status");
  statusEl.className = `status ${type}`;
  statusEl.textContent =
    message || type.charAt(0).toUpperCase() + type.slice(1);
}

function isValidWebSocketUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    return parsed.protocol === "ws:" || parsed.protocol === "wss:";
  } catch {
    return false;
  }
}

async function startCaptions(): Promise<void> {
  const serverUrl = getElement<HTMLInputElement>("server-url").value.trim();
  const userToken = getElement<HTMLInputElement>("user-token").value.trim();
  const language = getElement<HTMLSelectElement>("language").value;
  const targetLanguage = getElement<HTMLSelectElement>("target-language").value;

  if (!serverUrl || !userToken) {
    setStatus("error", "Please fill in server URL and token");
    return;
  }

  if (!isValidWebSocketUrl(serverUrl)) {
    setStatus("error", "Server URL must start with ws:// or wss://");
    return;
  }

  await saveSettings();

  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab || !tab.id) {
    setStatus("error", "No active tab");
    return;
  }

  try {
    await chrome.runtime.sendMessage({
      type: "START_CAPTIONS",
      serverUrl,
      userToken,
      language,
      targetLanguage,
      tabId: tab.id,
      settings: defaultSettings,
    });

    setStatus("connected", "Connecting...");
    getElement<HTMLButtonElement>("start-btn").disabled = true;
    getElement<HTMLButtonElement>("stop-btn").disabled = false;
  } catch (error) {

    const errorMessage =
      error instanceof Error ? error.message : "Unknown error";
    setStatus("error", `Failed to start: ${errorMessage}`);
  }
}

async function stopCaptions(): Promise<void> {
  try {
    await chrome.runtime.sendMessage({
      type: "STOP_CAPTIONS",
    });

    setStatus("disconnected", "Stopped");
    getElement<HTMLButtonElement>("start-btn").disabled = false;
    getElement<HTMLButtonElement>("stop-btn").disabled = true;
    isConnected = false;
  } catch (error) {
    const errorMessage =
      error instanceof Error ? error.message : "Unknown error";
    setStatus("error", `Failed to stop: ${errorMessage}`);
  }
}

function applySettings(): void {
  saveSettings();
  chrome.runtime.sendMessage({
    type: "UPDATE_SETTINGS",
    settings: defaultSettings,
  });
}

  document.addEventListener("DOMContentLoaded", async () => {
    await loadSettings();

    getElement("start-btn").addEventListener("click", startCaptions);
    getElement("stop-btn").addEventListener("click", stopCaptions);

    getElement("disable-current-video").addEventListener("change", async () => {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (!tab?.id) {
        return;
      }
      const key = await chrome.runtime.sendMessage({
        type: "GET_VIDEO_KEY",
        tabId: tab.id,
      });
      if (typeof key !== "string") {
        return;
      }
      const storageKey = `disabled:${key}`;
      const checked = getElement<HTMLInputElement>(
        "disable-current-video",
      ).checked;
      await chrome.storage.sync.set({ [storageKey]: checked });
    });

    getElement("font-size").addEventListener("input", () => {
      defaultSettings.fontSize = parseInt(
        getElement<HTMLInputElement>("font-size").value,
      );
      updateDisplayValues();
      applySettings();
    });


  getElement("background-opacity").addEventListener("input", () => {
    defaultSettings.backgroundOpacity = parseInt(
      getElement<HTMLInputElement>("background-opacity").value,
    );
    updateDisplayValues();
    applySettings();
  });

  getElement("position").addEventListener("change", () => {
    defaultSettings.position = getElement<HTMLSelectElement>("position")
      .value as "top" | "bottom";
    applySettings();
  });

  getElement("max-lines").addEventListener("change", () => {
    defaultSettings.maxLines = parseInt(
      getElement<HTMLSelectElement>("max-lines").value,
    );
    applySettings();
  });

  getElement("target-language").addEventListener("change", () => {
    defaultSettings.targetLanguage = getElement<HTMLSelectElement>(
      "target-language",
    ).value;
    saveSettings();
  });

  getElement("mute-captions").addEventListener("change", () => {
    defaultSettings.captionsMuted = getElement<HTMLInputElement>(
      "mute-captions",
    ).checked;
    applySettings();
  });

  chrome.runtime.onMessage.addListener((message: StatusUpdateMessage) => {
    if (message.type === "STATUS_UPDATE") {
      setStatus(message.status, message.message);
      isConnected = message.status === "connected";

      getElement<HTMLButtonElement>("start-btn").disabled = isConnected;
      getElement<HTMLButtonElement>("stop-btn").disabled = !isConnected;
    }
  });
});
