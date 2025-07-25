// Define patterns for YouTube requests we want to intercept
const youtubePatterns = [
  "*://www.youtube.com/youtubei/v1/next*",
  "*://www.youtube.com/youtubei/v1/browse*"
];

// Response modification settings
let responseModificationEnabled = false;
let serverUrl = "https://misilelab--himari-api-fastapi-app.modal.run/";
let apiKey = "";

// Store predefined response modification functions
const predefinedFunctions = {
  markBotComments: function(response, url) {
    try {
      if (!url.includes('/youtubei/v1/next') && !url.includes('/youtubei/v1/browse')) {
        return response;
      }
      
      if (!response || !response.frameworkUpdates || 
          !response.frameworkUpdates.entityBatchUpdate || 
          !response.frameworkUpdates.entityBatchUpdate.mutations) {
        return response;
      }
      
      const {mutations} = response.frameworkUpdates.entityBatchUpdate;
      
      for (const mutation of mutations) {
        if (mutation.payload && mutation.payload.commentEntityPayload) {
          const payload = mutation.payload.commentEntityPayload;
          
          if (payload.author && payload.author.displayName && 
              payload.properties && payload.properties.content) {
            const content = payload.properties.content.content.toLowerCase();
            if (content.includes('subscribe to my channel') || 
                content.includes('check out my video')) {
              payload.author.displayName += " [BOT]";
            }
          }
        }
      }
      
      return response;
    } catch (error) {
      console.error('Error in markBotComments function:', error);
      return response;
    }
  },
  
  // Add more predefined functions as needed
  removeSpamLinks: function(response, url) {
    try {
      if (!url.includes('/youtubei/v1/next') && !url.includes('/youtubei/v1/browse')) {
        return response;
      }
      
      if (!response || !response.frameworkUpdates || 
          !response.frameworkUpdates.entityBatchUpdate || 
          !response.frameworkUpdates.entityBatchUpdate.mutations) {
        return response;
      }
      
      const {mutations} = response.frameworkUpdates.entityBatchUpdate;
      
      for (const mutation of mutations) {
        if (mutation.payload && mutation.payload.commentEntityPayload) {
          const payload = mutation.payload.commentEntityPayload;
          
          if (payload.properties && payload.properties.content) {
            const {content} = payload.properties.content;
            // Remove URLs from comments
            payload.properties.content.content = content.replace(/https?:\/\/\S+/g, '[link removed]');
          }
        }
      }
      
      return response;
    } catch (error) {
      console.error('Error in removeSpamLinks function:', error);
      return response;
    }
  }
};

// Current active function
let activeModificationFunction = predefinedFunctions.markBotComments;

// Function to dynamically modify a response by fetching from example server
async function fetchModifiedResponse(originalUrl, requestId) {
  try {
    // Only attempt if modification is enabled and we have a server URL
    if (!responseModificationEnabled || !serverUrl) {
      return null;
    }

    // Prepare the URL for the example server
    const fetchUrl = `${serverUrl}/evaluate`;
    
    console.log(`Fetching modified response from: ${fetchUrl}`);
    
    // Prepare headers
    const headers = {
      'Content-Type': 'application/json'
    };
    
    // Fetch from the example server
    const response = await fetch(fetchUrl, {
      method: 'POST', // Changed from GET to POST for proper JSON body
      headers: headers,
      body: JSON.stringify({
        originalUrl: originalUrl,
        api_key: apiKey
      })
    });
    
    if (!response.ok) {
      throw new Error(`Server responded with status: ${response.status}`);
    }
    
    // Get the response data
    const responseData = await response.json();
    
    // Update our intercepted request with the modification info
    const index = interceptedRequests.findIndex(req => req.requestId === requestId);
    if (index !== -1) {
      interceptedRequests[index].modified = true;
      interceptedRequests[index].modifiedTimestamp = new Date().toISOString();
      interceptedRequests[index].modificationSource = 'server';
      interceptedRequests[index].processing = false;
      
      // Store the updated requests
      chrome.storage.local.set({ 'interceptedRequests': interceptedRequests });
    }
    
    return responseData;
  } catch (error) {
    console.error('Error fetching modified response:', error);
    
    // Update the request with the error if we can find it
    const index = interceptedRequests.findIndex(req => req.requestId === requestId);
    if (index !== -1) {
      interceptedRequests[index].modificationError = error.message;
      interceptedRequests[index].processing = false;
      chrome.storage.local.set({ 'interceptedRequests': interceptedRequests });
    }
    
    return null;
  }
}

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
    
    const comments = [];
    const {mutations} = responseData.frameworkUpdates.entityBatchUpdate;
    
    for (const mutation of mutations) {
      if (mutation.payload && mutation.payload.commentEntityPayload) {
        const payload = mutation.payload.commentEntityPayload;
        
        if (payload.author && payload.author.displayName && 
            payload.properties && payload.properties.content && 
            payload.properties.content.content) {
          const author = payload.author.displayName;
          const {content} = payload.properties.content;
          
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
    const response = await fetch(`${serverUrl}/evaluate`, {
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
        // Mark the request as being processed for modification
        if (index !== -1) {
          interceptedRequests[index].processing = true;
          interceptedRequests[index].modified = true;
          chrome.storage.local.set({ 'interceptedRequests': interceptedRequests });
        }

        // Start the dynamic response modification process
        // Note: In Manifest V3, we can't actually modify the response body here
        // This is just to track that we would modify this response
        fetchModifiedResponse(details.url, details.requestId);
      } catch (error) {
        console.error('Error in response modification:', error);
      }
    }
  },
  { urls: youtubePatterns },
  ["responseHeaders"]
);

// Add a declarative net request rule handler to apply our modifications
chrome.declarativeNetRequest.onRuleMatchedDebug?.addListener((info) => {
  console.log('Rule matched:', info);
  
  // Get the request details
  const { request, rule } = info;
  
  // If we have an active modification function and response modification is enabled
  if (responseModificationEnabled && activeModificationFunction) {
    try {
      console.log(`Applying ${activeModificationFunction.name} to response for ${request.url}`);
      
      // Find the request in our array
      const index = interceptedRequests.findIndex(req => req.url === request.url);
      if (index !== -1) {
        // Mark as modified by function
        interceptedRequests[index].modified = true;
        interceptedRequests[index].modificationSource = 'function';
        interceptedRequests[index].modifiedTimestamp = new Date().toISOString();
        
        // Update storage
        chrome.storage.local.set({ 'interceptedRequests': interceptedRequests });
      }
    } catch (error) {
      console.error('Error applying modification function:', error);
    }
  }
});

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
  
  if (message.action === 'applyModificationFunction') {
    try {
      // Check if we have an active modification function and response modification is enabled
      if (!responseModificationEnabled || !activeModificationFunction) {
        sendResponse({ success: false, error: 'Response modification is disabled or no function selected' });
        return true;
      }
      
      // Apply the active modification function to the response data
      const modifiedResponse = activeModificationFunction(message.responseData, message.url);
      
      // Send back the modified response
      sendResponse({ 
        success: true, 
        modifiedResponse: modifiedResponse,
        functionName: activeModificationFunction.name
      });
    } catch (error) {
      console.error('Error applying modification function:', error);
      sendResponse({ 
        success: false, 
        error: error.message 
      });
    }
    return true;
  }
  
  if (message.action === 'setResponseModification') {
    try {
      responseModificationEnabled = message.enabled;
      
      if (message.functionCode) {
        // Instead of using Function constructor, use predefined functions
        const functionName = message.functionCode.trim();
        if (predefinedFunctions[functionName]) {
          activeModificationFunction = predefinedFunctions[functionName];
          
          // Save the function name for later retrieval
          chrome.storage.local.set({ 'responseModificationFunctionName': functionName });
        } else {
          // If not a predefined function, use the default
          activeModificationFunction = predefinedFunctions.markBotComments;
          chrome.storage.local.set({ 'responseModificationFunctionName': 'markBotComments' });
        }
      } else {
        activeModificationFunction = null;
      }
      
      // Save the server URL if provided
      if (message.serverUrl !== undefined) {
        serverUrl = message.serverUrl;
        chrome.storage.local.set({ 'serverUrl': serverUrl });
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
        serverUrl: serverUrl,
        apiKey: apiKey,
        functionName: activeModificationFunction ? activeModificationFunction.name : null
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
      hasFunction: activeModificationFunction !== null,
      serverUrl: serverUrl,
      apiKey: apiKey
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
          sendResponse({ success: false, status: response.status, error: 'Server returned an error' });
        }
      })
      .catch(error => {
        sendResponse({ success: false, error: error.message });
      });
    
    return true; // Keep the message channel open for async response
  }
  
  if (message.action === 'testBotDetectionAPI') {
    const url = message.serverUrl || serverUrl;
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
  
  if (message.action === 'getPredefinedFunctions') {
    // Return a list of available predefined functions
    const functionNames = Object.keys(predefinedFunctions);
    sendResponse({ success: true, functions: functionNames });
    return true;
  }
});

// Initialize by loading any previously stored settings
chrome.storage.local.get([
  'interceptedRequests',
  'responseModificationEnabled', 
  'responseModificationFunctionName', 
  'serverUrl', 
  'apiKey', 
  'modifyApiKey'
], (data) => {
  if (data.interceptedRequests) {
    interceptedRequests = data.interceptedRequests;
  }
  
  if (data.responseModificationEnabled !== undefined) {
    responseModificationEnabled = data.responseModificationEnabled;
  }
  
  if (data.responseModificationFunctionName && predefinedFunctions[data.responseModificationFunctionName]) {
    activeModificationFunction = predefinedFunctions[data.responseModificationFunctionName];
  }
  
  if (data.serverUrl) {
    serverUrl = data.serverUrl;
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