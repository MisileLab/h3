// Bot detection settings from storage, with defaults
let botDetectionEnabled = true;
let serverUrl = "https://misilelab--himari-api-fastapi-app.modal.run";
let apiKey = "";
let markBotNames = true;
let enableReportMenu = true;

// Function to detect bots using the API
async function detectBots(comments) {
  console.log("🤖 detectBots called with", comments ? comments.length : 0, "comments");
  
  if (!comments || comments.length === 0) {
    console.log("⚠️ No comments to process");
    return null;
  }
  
  if (!botDetectionEnabled) {
    console.log("⚠️ Bot detection is disabled.");
    return null;
  }
  
  if (!serverUrl) {
    console.error("⛔ Server URL not configured.");
    return null;
  }
  
  if (!apiKey) {
    console.error("⛔ API key not configured.");
    return null;
  }
  
  try {
    const apiUrl = serverUrl.endsWith('/') ? `${serverUrl}evaluate` : `${serverUrl}/evaluate`;
    console.log(`🚀 Sending ${comments.length} comments to bot detection API: ${apiUrl}`);
    console.log(comments)
    
    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        evaluate: comments,
        api_key: apiKey
      })
    });
    
    console.log(`📨 API response status: ${response.status} ${response.statusText}`);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error(`❌ API error: ${response.status}, ${errorText}`);
      throw new Error(`API responded with status: ${response.status}, ${errorText}`);
    }
    
    const result = await response.json();
    console.log("✅ Bot detection API response:", result);
    return result;
    
  } catch (error) {
    console.error('⚠️ Error detecting bots:', error);
    return null;
  }
}

// Listen for messages from content script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log(`📨 Received message: action=${message.action}`);
  
  if (message.action === 'detectBots') {
    if (!message.comments || !Array.isArray(message.comments)) {
      console.error(`❌ Invalid comments data received.`);
      sendResponse({ success: false, error: 'Invalid comments data' });
      return true;
    }
    
    detectBots(message.comments)
      .then(results => {
        console.log(`✅ Bot detection successful, sending results back to content script`);
        sendResponse({ success: true, results: results });
      })
      .catch(error => {
        console.error(`❌ Bot detection failed: ${error.message}`);
        sendResponse({ success: false, error: error.message });
      });
    
    return true; // Keep the message channel open for async response
  }
  
  if (message.action === 'getBotDetectionSettings') {
    sendResponse({
      enabled: botDetectionEnabled,
      serverUrl: serverUrl,
      apiKey: apiKey,
      markBotNames: markBotNames,
      enableReportMenu: enableReportMenu
    });
    return true;
  }
  
  if (message.action === 'setBotDetectionSettings') {
    const settingsToUpdate = {};
    if (message.enabled !== undefined) {
      botDetectionEnabled = message.enabled;
      settingsToUpdate.botDetectionEnabled = botDetectionEnabled;
    }
    if (message.serverUrl) {
      serverUrl = message.serverUrl;
      settingsToUpdate.serverUrl = serverUrl;
    }
    if (message.apiKey !== undefined) {
      apiKey = message.apiKey;
      settingsToUpdate.apiKey = apiKey;
    }
    if (message.markBotNames !== undefined) {
      markBotNames = message.markBotNames;
      settingsToUpdate.markBotNames = markBotNames;
    }
    if (message.enableReportMenu !== undefined) {
      enableReportMenu = message.enableReportMenu;
      settingsToUpdate.enableReportMenu = enableReportMenu;
    }

    chrome.storage.local.set(settingsToUpdate, () => {
      if (chrome.runtime.lastError) {
        console.error('Error saving settings:', chrome.runtime.lastError.message);
        sendResponse({ success: false, error: chrome.runtime.lastError.message });
      } else {
        console.log('✅ Settings saved successfully:', settingsToUpdate);
        sendResponse({ success: true });
      }
    });
    return true;
  }
  
  if (message.action === 'testServerConnection') {
    const url = message.serverUrl || serverUrl;
    fetch(`${url}/docs`, { method: 'GET' })
      .then(response => {
        if (response.ok) {
          sendResponse({ success: true, status: response.status });
        } else {
          response.text().then(text => {
            sendResponse({ success: false, status: response.status, error: text });
          });
        }
      })
      .catch(error => {
        sendResponse({ success: false, error: error.message });
      });
    return true;
  }
  
  if (message.action === 'reportComment') {
    const url = message.serverUrl || serverUrl;
    const key = message.apiKey || apiKey;
    const isBot = message.isBot !== undefined ? message.isBot : true;
    
    if (!key || !message.comment) {
      sendResponse({ success: false, error: 'API key and comment data are required' });
      return true;
    }
    
    fetch(`${url}/report`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        author_name: message.comment.author_name,
        content: message.comment.content,
        is_bot: isBot,
        api_key: key
      })
    })
    .then(response => response.json())
    .then(data => sendResponse({ success: true, data: data }))
    .catch(error => sendResponse({ success: false, error: error.message }));
    
    return true;
  }
});

// Initialize by loading settings from storage
function loadSettings() {
  chrome.storage.local.get([
    'botDetectionEnabled',
    'serverUrl',
    'apiKey',
    'markBotNames',
    'enableReportMenu'
  ], (data) => {
    if (chrome.runtime.lastError) {
      console.error("Error loading settings:", chrome.runtime.lastError.message);
      return;
    }
    if (data.botDetectionEnabled !== undefined) botDetectionEnabled = data.botDetectionEnabled;
    if (data.serverUrl) serverUrl = data.serverUrl;
    if (data.apiKey) apiKey = data.apiKey;
    if (data.markBotNames !== undefined) markBotNames = data.markBotNames;
    if (data.enableReportMenu !== undefined) enableReportMenu = data.enableReportMenu;
    console.log("🔧 Settings loaded:", { botDetectionEnabled, serverUrl, apiKey: apiKey ? 'set' : 'not set', markBotNames, enableReportMenu });
    setupContextMenu(); // Set up the context menu based on loaded settings
  });
}

// Function to set up or remove the context menu
function setupContextMenu() {
  chrome.contextMenus.removeAll(() => {
    if (enableReportMenu) {
      chrome.contextMenus.create({
        id: 'reportAsBot',
        title: 'Report as Bot',
        contexts: ['all'],
        documentUrlPatterns: ['*://www.youtube.com/*']
      });
      chrome.contextMenus.create({
        id: 'reportAsUser',
        title: 'Report as User',
        contexts: ['all'],
        documentUrlPatterns: ['*://www.youtube.com/*']
      });
    }
  });
}

// Update context menu when the setting changes
chrome.storage.onChanged.addListener((changes, namespace) => {
  if (changes.enableReportMenu) {
    enableReportMenu = changes.enableReportMenu.newValue;
    setupContextMenu();
  }
});

// Listen for context menu clicks
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === 'reportAsBot' || info.menuItemId === 'reportAsUser') {
    chrome.tabs.sendMessage(tab.id, {
      action: 'contextMenuClick',
      menuItemId: info.menuItemId
    });
  }
});

// Load settings on startup and set up the context menu
loadSettings();

console.log('YouTube Bot Detector background script loaded and simplified.'); 