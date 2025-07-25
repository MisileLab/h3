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
    
    // Fetch from the example server
    const response = await fetch(fetchUrl, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        'X-Original-URL': originalUrl
      }
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
    // Execute the custom function
    return responseModificationFunction(response, url);
  } catch (error) {
    console.error('Error applying custom function:', error);
    return response;
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
    
    // If response modification is enabled and we have a function
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
      
      sendResponse({ 
        success: true, 
        enabled: responseModificationEnabled,
        serverUrl: exampleServerUrl
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
      serverUrl: exampleServerUrl
    });
    return true;
  }
  
  if (message.action === 'testServerConnection') {
    const url = message.serverUrl || exampleServerUrl;
    
    fetch(url, { method: 'GET' })
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
});

// Initialize by loading any previously stored requests
chrome.storage.local.get(['interceptedRequests', 'responseModificationEnabled', 'responseModificationFunction', 'exampleServerUrl'], (data) => {
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
});

console.log('YouTube Request Interceptor background script loaded'); 