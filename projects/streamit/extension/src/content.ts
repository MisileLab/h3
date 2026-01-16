import type { CaptionSettings, ContentScriptMessage } from './types';

const captionLines: string[] = [];

let settings: CaptionSettings = {
  fontSize: 24,
  backgroundOpacity: 70,
  position: 'bottom',
  maxLines: 2
};

function createOverlay(): HTMLDivElement {
  const overlay = document.createElement('div');
  overlay.id = 'streamit-overlay';
  document.body.appendChild(overlay);
  return overlay;
}

let overlayElement = createOverlay();

function updateOverlayPosition(): void {
  const overlay = overlayElement;

  if (settings.position === 'top') {
    overlay.style.top = '0';
    overlay.style.bottom = 'auto';
  } else {
    overlay.style.top = 'auto';
    overlay.style.bottom = '0';
  }

  overlay.style.fontSize = `${settings.fontSize}px`;
}

function addCaption(text: string, isPartial = false): void {
  if (!text || !text.trim()) {
    return;
  }

  if (isPartial) {
    if (captionLines.length > 0) {
      captionLines[captionLines.length - 1] = text;
    } else {
      captionLines.push(text);
    }
  } else {
    captionLines.push(text);
  }

  while (captionLines.length > settings.maxLines) {
    captionLines.shift();
  }

  renderCaptions();
}

function renderCaptions(): void {
  const overlay = overlayElement;
  overlay.innerHTML = '';

  const backgroundOpacity = settings.backgroundOpacity / 100;

  captionLines.forEach((text) => {
    const line = document.createElement('div');
    line.style.setProperty('--bg-opacity', String(backgroundOpacity));
    line.textContent = text;
    overlay.appendChild(line);
  });
}

function clearCaptions(): void {
  captionLines.length = 0;
  if (overlayElement) {
    overlayElement.innerHTML = '';
  }
}

function updateSettings(newSettings: Partial<CaptionSettings>): void {
  Object.assign(settings, newSettings);
  updateOverlayPosition();
  renderCaptions();
}

chrome.runtime.onMessage.addListener((message: ContentScriptMessage) => {
  switch (message.type) {
    case 'CAPTION_PARTIAL':
      addCaption(message.text, true);
      break;
    case 'CAPTION_FINAL':
      addCaption(message.text, false);
      break;
    case 'UPDATE_SETTINGS':
      updateSettings(message.settings);
      break;
    case 'CLEAR_CAPTIONS':
      clearCaptions();
      break;
  }
});

document.addEventListener('DOMContentLoaded', () => {
  updateOverlayPosition();
});
