// Define patterns for YouTube requests we want to intercept
const youtubePatterns = [
  "*://www.youtube.com/*",
  "*://*.youtube.com/api/*",
  "*://*.youtube.com/youtubei/*"
];

// Store intercepted requests
let interceptedRequests = [];

// Response modification settings
let responseModificationEnabled = false;
let responseModificationFunction = null;
let exampleServerUrl = "https://example.com/api"; // Default example server URL
let apiKey = ""; // API key for the bot detection service

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

// Function to dynamically modify a response by fetching from example server
async function fetchModifiedResponse(originalUrl, requestId) {
  try {
    // Only attempt if modification is enabled and we have a server URL
    if (!responseModificationEnabled || !exampleServerUrl) {
      return null;
    }

    // Find the request in our array
    const index = interceptedRequests.findIndex(req => req.requestId === requestId);
    if (index === -1) return null;

    // Mark as being processed
    interceptedRequests[index].processing = true;
    
    // Prepare the URL for the example server
    // We'll pass the original URL as a parameter
    const fetchUrl = `${exampleServerUrl}?originalUrl=${encodeURIComponent(originalUrl)}`;
    
    console.log(`Fetching modified response from: ${fetchUrl}`);
    
    // Prepare headers
    const headers = {
      'Content-Type': 'application/json',
      'X-Original-URL': originalUrl
    };
    
    // Add API key if available
    if (apiKey) {
      headers['Authorization'] = `Bearer ${apiKey}`;
    }
    
    // Fetch from the example server
    const response = await fetch(fetchUrl, {
      method: 'GET',
      headers: headers
    });
    
    if (!response.ok) {
      throw new Error(`Server responded with status: ${response.status}`);
    }
    
    // Get the response data
    const responseData = await response.json();
    
    // Update our intercepted request with the modification info
    interceptedRequests[index].modified = true;
    interceptedRequests[index].modifiedTimestamp = new Date().toISOString();
    interceptedRequests[index].modificationSource = 'server';
    interceptedRequests[index].processing = false;
    
    // Store the updated requests
    chrome.storage.local.set({ 'interceptedRequests': interceptedRequests });
    
    return responseData;
  } catch (error) {
    console.error('Error fetching modified response:', error);
    
    // Update the request with the error
    if (index !== -1) {
      interceptedRequests[index].modificationError = error.message;
      interceptedRequests[index].processing = false;
      chrome.storage.local.set({ 'interceptedRequests': interceptedRequests });
    }
    
    return null;
  }
}

// Function to apply custom function to response
function applyCustomFunction(response, url) {
  if (!responseModificationFunction) return response;
  
  try {
    return responseModificationFunction(response, url);
  } catch (error) {
    console.error('Error applying custom function:', error);
    return response;
  }
}

// Function to extract comments from YouTube response
function extractCommentsFromResponse(responseData) {
  try {
    if (!responseData || !responseData.frameworkUpdates || 
        !responseData.frameworkUpdates.entityBatchUpdate || 
        !responseData.frameworkUpdates.entityBatchUpdate.mutations) {
      return null;
    }
    
    const comments = [];
    const mutations = responseData.frameworkUpdates.entityBatchUpdate.mutations;
    
    for (const mutation of mutations) {
      if (mutation.payload && mutation.payload.commentEntityPayload) {
        const payload = mutation.payload.commentEntityPayload;
        
        if (payload.author && payload.author.displayName && 
            payload.properties && payload.properties.content && 
            payload.properties.content.content) {
          const author = payload.author.displayName;
          const content = payload.properties.content.content;
          
          comments.push({ author_name: author, content: content });
        }
      }
    }
    
    return comments.length > 0 ? comments : null;
  } catch (error) {
    console.error('Error extracting comments:', error);
    return null;
  }
}

// Function to send comments to bot detection API
async function detectBots(comments) {
  if (!comments || comments.length === 0 || !apiKey) {
    return null;
  }
  
  try {
    const response = await fetch(exampleServerUrl + '/evaluate', {
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
      throw new Error(`API responded with status: ${response.status}`);
    }
    
    return await response.json();
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
    
    const mutations = responseData.frameworkUpdates.entityBatchUpdate.mutations;
    let commentIndex = 0;
    
    for (let i = 0; i < mutations.length; i++) {
      const mutation = mutations[i];
      if (mutation.payload && mutation.payload.commentEntityPayload) {
        const payload = mutation.payload.commentEntityPayload;
        
        if (payload.author && payload.author.displayName && 
            commentIndex < botDetectionResults.is_bot.length) {
          
          // If the API identifies this as a bot, change the author name
          if (botDetectionResults.is_bot[commentIndex]) {
            payload.author.displayName = payload.author.displayName + " [BOT]";
            
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
    const modifiedResponse = modifyResponseWithBotDetection(responseData, botDetectionResults);
    
    return modifiedResponse;
  } catch (error) {
    console.error('Error processing YouTube comment response:', error);
    return responseData;
  }
}

// Set up the request interception (non-blocking in Manifest V3)
chrome.webRequest.onBeforeRequest.addListener(
  interceptRequest,
  { urls: youtubePatterns }
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
    
    // If response modification is enabled
    if (responseModificationEnabled) {
      try {
        // Start the dynamic response modification process
        // Note: In Manifest V3, we can't actually modify the response body here
        // This is just to track that we would modify this response
        fetchModifiedResponse(details.url, details.requestId);
        
        interceptedRequests[index].modified = true;
        chrome.storage.local.set({ 'interceptedRequests': interceptedRequests });
      } catch (error) {
        console.error('Error in response modification:', error);
      }
    }
  },
  { urls: youtubePatterns },
  ["responseHeaders"]
);

// Also listen for completed requests to get response data
chrome.webRequest.onCompleted.addListener(
  (details) => {
    // Find the matching request in our array
    const index = interceptedRequests.findIndex(req => req.requestId === details.requestId);
    if (index !== -1) {
      // Update with response info
      interceptedRequests[index].statusCode = details.statusCode;
      interceptedRequests[index].responseTime = new Date().toISOString();
      
      // Update storage
      chrome.storage.local.set({ 'interceptedRequests': interceptedRequests });
    }
  },
  { urls: youtubePatterns }
);

// Listen for messages from content script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'getInterceptedRequests') {
    sendResponse({ requests: interceptedRequests });
    return true; // Keep the message channel open for async response
  }
  
  if (message.action === 'setResponseModification') {
    try {
      responseModificationEnabled = message.enabled;
      
      if (message.functionCode) {
        // Create a function from the string code
        responseModificationFunction = new Function('response', 'url', message.functionCode);
      } else {
        responseModificationFunction = null;
      }
      
      // Save the server URL if provided
      if (message.serverUrl !== undefined) {
        exampleServerUrl = message.serverUrl;
        chrome.storage.local.set({ 'exampleServerUrl': exampleServerUrl });
      }
      
      // Save the API key if provided
      if (message.apiKey !== undefined) {
        apiKey = message.apiKey;
        chrome.storage.local.set({ 'apiKey': apiKey });
      }
      
      // Save the mark bot names setting if provided
      if (message.markBotNames !== undefined) {
        chrome.storage.local.set({ 'markBotNames': message.markBotNames });
      }
      
      // Save the enable report menu setting if provided
      if (message.enableReportMenu !== undefined) {
        chrome.storage.local.set({ 'enableReportMenu': message.enableReportMenu });
      }
      
      sendResponse({ 
        success: true, 
        enabled: responseModificationEnabled,
        serverUrl: exampleServerUrl,
        apiKey: apiKey
      });
    } catch (error) {
      sendResponse({ 
        success: false, 
        error: error.message 
      });
    }
    return true;
  }
  
  if (message.action === 'getResponseModificationStatus') {
    sendResponse({
      enabled: responseModificationEnabled,
      hasFunction: responseModificationFunction !== null,
      serverUrl: exampleServerUrl,
      apiKey: apiKey
    });
    return true;
  }
  
  if (message.action === 'testServerConnection') {
    const url = message.serverUrl || exampleServerUrl;
    const key = message.apiKey || apiKey;
    
    // Prepare headers
    const headers = {
      'Content-Type': 'application/json'
    };
    
    // Add API key if available
    if (key) {
      headers['Authorization'] = `Bearer ${key}`;
    }
    
    fetch(url, { 
      method: 'GET',
      headers: headers
    })
      .then(response => {
        if (response.ok) {
          sendResponse({ success: true, status: response.status });
        } else {
          sendResponse({ success: false, status: response.status, error: 'Server returned an error' });
        }
      })
      .catch(error => {
        sendResponse({ success: false, error: error.message });
      });
    
    return true; // Keep the message channel open for async response
  }
  
  if (message.action === 'testBotDetectionAPI') {
    const url = message.serverUrl || exampleServerUrl;
    const key = message.apiKey || apiKey;
    
    if (!key) {
      sendResponse({ success: false, error: 'API key is required' });
      return true;
    }
    
    const testComment = {
      author_name: "Test User",
      content: "This is a test comment."
    };
    
    fetch(`${url}/evaluate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        evaluate: [testComment],
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
  
  if (message.action === 'reportComment') {
    const url = message.serverUrl || exampleServerUrl;
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

// Initialize by loading any previously stored requests
chrome.storage.local.get(['interceptedRequests', 'responseModificationEnabled', 'responseModificationFunction', 'exampleServerUrl', 'apiKey', 'modifyApiKey'], (data) => {
  if (data.interceptedRequests) {
    interceptedRequests = data.interceptedRequests;
  }
  
  if (data.responseModificationEnabled !== undefined) {
    responseModificationEnabled = data.responseModificationEnabled;
  }
  
  if (data.responseModificationFunction) {
    try {
      responseModificationFunction = new Function('response', 'url', data.responseModificationFunction);
    } catch (error) {
      console.error('Error recreating response modification function:', error);
    }
  }
  
  if (data.exampleServerUrl) {
    exampleServerUrl = data.exampleServerUrl;
  }
  
  if (data.apiKey) {
    apiKey = data.apiKey;
  }
  
  // If there's a specific API key for response modification, use that instead
  if (data.modifyApiKey) {
    apiKey = data.modifyApiKey;
  }
});

console.log('YouTube Request Interceptor background script loaded'); 