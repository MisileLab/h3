// Define patterns for YouTube requests we want to intercept
const youtubePatterns = [
  "*://www.youtube.com/youtubei/v1/next*",
  "*://www.youtube.com/youtubei/v1/browse*"
];

// Bot detection settings
let botDetectionEnabled = true;
let serverUrl = "https://misilelab--himari-api-fastapi-app.modal.run";
let apiKey = "";
let markBotNames = true;
let enableReportMenu = true;

// Store intercepted requests
let interceptedRequests = [];

// Function to intercept requests
function interceptRequest(details) {
  // Skip non-XHR requests
  if (details.type !== 'xmlhttprequest') return;

  // Store request details
  const requestInfo = {
    url: details.url,
    method: details.method,
    timestamp: new Date().toISOString(),
    type: details.type,
    requestId: details.requestId
  };
  
  interceptedRequests.push(requestInfo);
  
  // Keep only the last 100 requests to avoid memory issues
  if (interceptedRequests.length > 100) {
    interceptedRequests.shift();
  }
  
  // Store in local storage
  chrome.storage.local.set({ 'interceptedRequests': interceptedRequests });
  
  // Log the intercepted request
  console.log('Intercepted YouTube request:', requestInfo);
}

// Function to extract comments from YouTube response
function extractCommentsFromResponse(responseData) {
  try {
    if (!responseData || !responseData.frameworkUpdates || 
        !responseData.frameworkUpdates.entityBatchUpdate || 
        !responseData.frameworkUpdates.entityBatchUpdate.mutations) {
      return null;
    }
    
    const {mutations} = responseData.frameworkUpdates.entityBatchUpdate;
    const comments = [];
    
    for (const mutation of mutations) {
      if (mutation.payload && mutation.payload.commentEntityPayload) {
        const payload = mutation.payload.commentEntityPayload;
        
        if (payload.author && payload.author.displayName && 
            payload.properties && payload.properties.content) {
          comments.push({
            author: payload.author.displayName,
            content: payload.properties.content.content
          });
        }
      }
    }
    
    return comments.length > 0 ? comments : null;
  } catch (error) {
    console.error('Error extracting comments:', error);
    return null;
  }
}

// Function to detect bots using the API
async function detectBots(comments) {
  if (!comments || comments.length === 0) {
    console.log("No comments to detect bots in");
    return null;
  }
  
  // Check if bot detection is enabled and we have necessary settings
  if (!botDetectionEnabled) {
    console.log("Bot detection is disabled");
    return null;
  }
  
  if (!serverUrl) {
    console.error("Server URL not configured");
    return null;
  }
  
  try {
    // Ensure serverUrl doesn't end with a slash
    const apiUrl = serverUrl.endsWith('/') ? `${serverUrl}evaluate` : `${serverUrl}/evaluate`;
    
    console.log(`Sending ${comments.length} comments to bot detection API: ${apiUrl}`);
    console.log("API Key present:", !!apiKey);
    
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
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API responded with status: ${response.status}, ${errorText}`);
    }
    
    const result = await response.json();
    console.log("Bot detection API response:", result);
    return result;
  } catch (error) {
    console.error('Error detecting bots:', error);
    return null;
  }
}

// Function to modify YouTube response with bot detection results
function modifyResponseWithBotDetection(responseData, botDetectionResults) {
  if (!responseData || !botDetectionResults || 
      !botDetectionResults.is_bot || botDetectionResults.is_bot.length === 0) {
    return responseData;
  }
  
  try {
    if (!responseData.frameworkUpdates || 
        !responseData.frameworkUpdates.entityBatchUpdate || 
        !responseData.frameworkUpdates.entityBatchUpdate.mutations) {
      return responseData;
    }
    
    const {mutations} = responseData.frameworkUpdates.entityBatchUpdate;
    let commentIndex = 0;
    
    for (let i = 0; i < mutations.length; i++) {
      const mutation = mutations[i];
      if (mutation.payload && mutation.payload.commentEntityPayload) {
        const payload = mutation.payload.commentEntityPayload;
        
        if (payload.author && payload.author.displayName && 
            commentIndex < botDetectionResults.is_bot.length) {
          
          // If the API identifies this as a bot, change the author name
          if (botDetectionResults.is_bot[commentIndex]) {
            payload.author.displayName += " [BOT]";
            
            // Log the modification
            console.log(`Modified author name for bot: ${payload.author.displayName}`);
          }
          
          commentIndex++;
        }
      }
    }
    
    return responseData;
  } catch (error) {
    console.error('Error modifying response with bot detection:', error);
    return responseData;
  }
}

// Function to process YouTube comment responses
async function processYouTubeCommentResponse(responseData, url) {
  // Check if this is a comments response
  if (!url.includes('/youtubei/v1/next') && !url.includes('/youtubei/v1/browse')) {
    return responseData;
  }
  
  try {
    // Extract comments from the response
    const comments = extractCommentsFromResponse(responseData);
    if (!comments) {
      return responseData;
    }
    
    console.log(`Extracted ${comments.length} comments from response`);
    
    // Send comments to bot detection API
    const botDetectionResults = await detectBots(comments);
    if (!botDetectionResults) {
      return responseData;
    }
    
    console.log('Bot detection results:', botDetectionResults);
    
    // Modify the response with bot detection results
    return modifyResponseWithBotDetection(responseData, botDetectionResults);
  } catch (error) {
    console.error('Error processing YouTube comment response:', error);
    return responseData;
  }
}

// Listen for web requests to intercept
chrome.webRequest.onBeforeRequest.addListener(
  interceptRequest,
  { urls: youtubePatterns },
  []
);

// Listen for response headers to potentially modify them
chrome.webRequest.onHeadersReceived.addListener(
  (details) => {
    // Find the matching request in our array
    const index = interceptedRequests.findIndex(req => req.requestId === details.requestId);
    if (index !== -1) {
      // Update with response info
      interceptedRequests[index].statusCode = details.statusCode;
      interceptedRequests[index].responseHeaders = details.responseHeaders;
      interceptedRequests[index].responseTime = new Date().toISOString();
      
      // Update storage
      chrome.storage.local.set({ 'interceptedRequests': interceptedRequests });
    }
  },
  { urls: youtubePatterns },
  ["responseHeaders"]
);

// Listen for messages from content script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'getInterceptedRequests') {
    sendResponse({ requests: interceptedRequests });
    return true; // Keep the message channel open for async response
  }
  
  if (message.action === 'detectBots') {
    if (!message.comments || !Array.isArray(message.comments)) {
      sendResponse({ success: false, error: 'Invalid comments data' });
      return true;
    }
    
    detectBots(message.comments)
      .then(results => {
        sendResponse({ success: true, results: results });
      })
      .catch(error => {
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
    if (message.enabled !== undefined) {
      botDetectionEnabled = message.enabled;
      chrome.storage.local.set({ 'botDetectionEnabled': botDetectionEnabled });
    }
    
    if (message.serverUrl) {
      serverUrl = message.serverUrl;
      chrome.storage.local.set({ 'serverUrl': serverUrl });
    }
    
    if (message.apiKey !== undefined) {
      apiKey = message.apiKey;
      chrome.storage.local.set({ 'apiKey': apiKey });
    }
    
    if (message.markBotNames !== undefined) {
      markBotNames = message.markBotNames;
      chrome.storage.local.set({ 'markBotNames': markBotNames });
    }
    
    if (message.enableReportMenu !== undefined) {
      enableReportMenu = message.enableReportMenu;
      chrome.storage.local.set({ 'enableReportMenu': enableReportMenu });
    }
    
    sendResponse({
      success: true,
      enabled: botDetectionEnabled,
      serverUrl: serverUrl,
      apiKey: apiKey,
      markBotNames: markBotNames,
      enableReportMenu: enableReportMenu
    });
    return true;
  }
  
  if (message.action === 'testServerConnection') {
    const url = message.serverUrl || serverUrl;
    
    // Prepare headers
    const headers = {
      'Content-Type': 'application/json'
    };
    
    fetch(`${url}/docs`, { 
      method: 'GET',
      headers: headers
    })
      .then(response => {
        if (response.ok) {
          sendResponse({ success: true, status: response.status });
        } else {
          return response.text().then(text => {
            sendResponse({ success: false, status: response.status, error: text });
          });
        }
      })
      .catch(error => {
        sendResponse({ success: false, error: error.message });
      });
    
    return true; // Keep the message channel open for async response
  }
  
  if (message.action === 'reportComment') {
    const url = message.serverUrl || serverUrl;
    const key = message.apiKey || apiKey;
    
    if (!key) {
      sendResponse({ success: false, error: 'API key is required' });
      return true;
    }
    
    if (!message.comment) {
      sendResponse({ success: false, error: 'Comment data is required' });
      return true;
    }
    
    fetch(`${url}/report`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        author_name: message.comment.author_name,
        content: message.comment.content,
        is_bot: true,
        api_key: key
      })
    })
      .then(response => {
        if (response.ok) {
          return response.json().then(data => {
            sendResponse({ success: true, data: data });
          });
        } else {
          return response.text().then(text => {
            sendResponse({ success: false, status: response.status, error: text });
          });
        }
      })
      .catch(error => {
        sendResponse({ success: false, error: error.message });
      });
    
    return true; // Keep the message channel open for async response
  }
});

// Initialize by loading any previously stored settings
chrome.storage.local.get([
  'interceptedRequests',
  'botDetectionEnabled',
  'serverUrl',
  'apiKey',
  'markBotNames',
  'enableReportMenu'
], (data) => {
  if (data.interceptedRequests) {
    interceptedRequests = data.interceptedRequests;
  }
  
  if (data.botDetectionEnabled !== undefined) {
    botDetectionEnabled = data.botDetectionEnabled;
  }
  
  if (data.serverUrl) {
    serverUrl = data.serverUrl;
  }
  
  if (data.apiKey) {
    apiKey = data.apiKey;
  }
  
  if (data.markBotNames !== undefined) {
    markBotNames = data.markBotNames;
  }
  
  if (data.enableReportMenu !== undefined) {
    enableReportMenu = data.enableReportMenu;
  }
});

// Set up context menu for reporting comments
chrome.contextMenus.create({
  id: 'reportBotComment',
  title: 'Report as Bot',
  contexts: ['all'],
  documentUrlPatterns: ['*://www.youtube.com/*']
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === 'reportBotComment') {
    // The actual reporting is handled by the content script
    chrome.tabs.sendMessage(tab.id, { action: 'contextMenuReportComment' });
  }
});

console.log('YouTube Bot Detector background script loaded'); 