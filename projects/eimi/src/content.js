// Add a control panel to the YouTube page
function addControlPanel() {
  // Create the panel container
  const panel = document.createElement('div');
  panel.id = 'yt-bot-detector-panel';
  panel.style.cssText = `
    position: fixed;
    bottom: 20px;
    right: 20px;
    width: 300px;
    background-color: rgba(33, 33, 33, 0.9);
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    z-index: 9999;
    font-family: 'Roboto', Arial, sans-serif;
    color: #fff;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  `;
  
  // Create the header
  const header = document.createElement('div');
  header.style.cssText = `
    padding: 10px;
    background-color: #282828;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #444;
  `;
  
  const title = document.createElement('div');
  title.textContent = 'YouTube Bot Detector';
  title.style.fontWeight = 'bold';
  
  const closeBtn = document.createElement('button');
  closeBtn.innerHTML = '&times;';
  closeBtn.style.cssText = `
    background: none;
    border: none;
    color: #aaa;
    font-size: 20px;
    cursor: pointer;
    padding: 0 5px;
  `;
  closeBtn.addEventListener('click', () => {
    panel.style.display = 'none';
  });
  
  header.appendChild(title);
  header.appendChild(closeBtn);
  
  // Create the tabs
  const tabsContainer = document.createElement('div');
  tabsContainer.style.cssText = `
    display: flex;
    background-color: #333;
  `;
  
  const botDetectionTab = document.createElement('div');
  botDetectionTab.textContent = 'Bot Detection';
  botDetectionTab.style.cssText = `
    padding: 8px 15px;
    cursor: pointer;
    background-color: #065fd4;
    color: white;
    flex: 1;
    text-align: center;
    font-size: 14px;
  `;
  
  tabsContainer.appendChild(botDetectionTab);
  
  // Create the content container
  const contentContainer = document.createElement('div');
  contentContainer.style.cssText = `
    padding: 15px;
    max-height: 400px;
    overflow-y: auto;
  `;
  
  // Create the bot detection container
  const botDetectionContainer = document.createElement('div');
  botDetectionContainer.id = 'yt-bot-detection-container';
  botDetectionContainer.innerHTML = `
    <div style="margin-bottom: 10px;">
      <label style="display: flex; align-items: center; margin-bottom: 10px;">
        <input type="checkbox" id="yt-bot-detection-enabled" style="margin-right: 10px;">
        <span>Enable Bot Detection</span>
      </label>
    </div>
    <div style="margin-bottom: 10px;">
      <div style="margin-bottom: 5px;">Bot Detection API URL:</div>
      <input type="text" id="yt-bot-api-url" placeholder="https://api.example.com" style="width: 100%; background: #222; color: #eee; border: 1px solid #444; padding: 5px; border-radius: 3px;">
    </div>
    <div style="margin-bottom: 10px;">
      <div style="margin-bottom: 5px;">API Key:</div>
      <div style="display: flex; align-items: center;">
        <input type="password" id="yt-api-key" placeholder="Your API Key" style="flex: 1; background: #222; color: #eee; border: 1px solid #444; padding: 5px; border-radius: 3px;">
        <button id="yt-toggle-api-key" style="background: #444; color: white; border: none; padding: 5px 8px; margin-left: 5px; border-radius: 3px; cursor: pointer;">👁️</button>
      </div>
    </div>
    <div style="margin-bottom: 10px;">
      <label style="display: flex; align-items: center; margin-bottom: 10px;">
        <input type="checkbox" id="yt-bot-mark-names" style="margin-right: 10px;">
        <span>Mark Bot Names with Icon</span>
      </label>
    </div>
    <div style="margin-bottom: 10px;">
      <label style="display: flex; align-items: center; margin-bottom: 10px;">
        <input type="checkbox" id="yt-enable-report-menu" style="margin-right: 10px;">
        <span>Enable Right-Click Report Menu</span>
      </label>
    </div>
    <div style="display: flex; margin-bottom: 10px;">
      <button id="yt-test-bot-api" style="background: #065fd4; color: white; border: none; padding: 5px 10px; border-radius: 3px; cursor: pointer; margin-right: 10px;">Test Connection</button>
      <button id="yt-save-bot-settings" style="background: #065fd4; color: white; border: none; padding: 5px 10px; border-radius: 3px; cursor: pointer;">Save Settings</button>
    </div>
    <div id="yt-bot-api-status" style="margin-top: 10px; font-size: 12px;"></div>
  `;
  
  contentContainer.appendChild(botDetectionContainer);
  
  // Add everything to the panel
  panel.appendChild(header);
  panel.appendChild(tabsContainer);
  panel.appendChild(contentContainer);
  
  // Add the panel to the page
  document.body.appendChild(panel);
  
  // Make the panel draggable
  makeDraggable(panel, header);
  
  // Set up bot detection
  setupBotDetection();
  
  // Return the panel for further use
  return { panel, contentContainer };
}

// Function to set up bot detection
function setupBotDetection() {
  setTimeout(() => {
    const enabledCheckbox = document.getElementById('yt-bot-detection-enabled');
    const apiKeyInput = document.getElementById('yt-api-key');
    const apiUrlInput = document.getElementById('yt-bot-api-url');
    const toggleApiKeyBtn = document.getElementById('yt-toggle-api-key');
    const testApiBtn = document.getElementById('yt-test-bot-api');
    const saveSettingsBtn = document.getElementById('yt-save-bot-settings');
    const apiStatusDiv = document.getElementById('yt-bot-api-status');
    const markNamesCheckbox = document.getElementById('yt-bot-mark-names');
    const enableReportMenuCheckbox = document.getElementById('yt-enable-report-menu');
    
    if (!enabledCheckbox || !apiKeyInput || !apiUrlInput || !toggleApiKeyBtn || 
        !testApiBtn || !saveSettingsBtn || !apiStatusDiv || !markNamesCheckbox || 
        !enableReportMenuCheckbox) return;
    
    // Toggle API key visibility
    toggleApiKeyBtn.addEventListener('click', () => {
      if (apiKeyInput.type === 'password') {
        apiKeyInput.type = 'text';
        toggleApiKeyBtn.textContent = '🔒';
      } else {
        apiKeyInput.type = 'password';
        toggleApiKeyBtn.textContent = '👁️';
      }
    });
    
    // Test API connection
    testApiBtn.addEventListener('click', () => {
      const serverUrl = apiUrlInput.value.trim();
      const apiKey = apiKeyInput.value.trim();
      
      if (!serverUrl) {
        apiStatusDiv.textContent = 'Error: Server URL is required';
        apiStatusDiv.style.color = '#f44336';
        return;
      }
      
      apiStatusDiv.textContent = 'Testing connection...';
      apiStatusDiv.style.color = '#ffeb3b';
      
      chrome.runtime.sendMessage({
        action: 'testServerConnection',
        serverUrl: serverUrl,
        apiKey: apiKey
      }, (response) => {
        if (response && response.success) {
          apiStatusDiv.textContent = 'Connection successful!';
          apiStatusDiv.style.color = '#4caf50';
        } else {
          const errorMsg = response && response.error ? response.error : 'Unknown error';
          apiStatusDiv.textContent = `Error: ${errorMsg}`;
          apiStatusDiv.style.color = '#f44336';
        }
        setTimeout(() => { apiStatusDiv.textContent = ''; }, 3000);
      });
    });
    
    // Save settings
    saveSettingsBtn.addEventListener('click', () => {
      const enabled = enabledCheckbox.checked;
      const serverUrl = apiUrlInput.value.trim();
      const apiKey = apiKeyInput.value.trim();
      const markBotNames = markNamesCheckbox.checked;
      const enableReportMenu = enableReportMenuCheckbox.checked;
      
      if (enabled && !serverUrl) {
        apiStatusDiv.textContent = 'Error: Server URL is required';
        apiStatusDiv.style.color = '#f44336';
        return;
      }
      
      chrome.runtime.sendMessage({
        action: 'setBotDetectionSettings',
        enabled: enabled,
        serverUrl: serverUrl,
        apiKey: apiKey,
        markBotNames: markBotNames,
        enableReportMenu: enableReportMenu
      }, (response) => {
        if (response && response.success) {
          apiStatusDiv.textContent = enabled ? 'Bot detection enabled!' : 'Bot detection disabled!';
          apiStatusDiv.style.color = '#4caf50';
          
          // Save settings to storage
          chrome.storage.local.set({
            'botDetectionEnabled': enabled,
            'serverUrl': serverUrl,
            'apiKey': apiKey,
            'markBotNames': markBotNames,
            'enableReportMenu': enableReportMenu
          });
        } else {
          const errorMsg = response && response.error ? response.error : 'Unknown error';
          apiStatusDiv.textContent = `Error: ${errorMsg}`;
          apiStatusDiv.style.color = '#f44336';
        }
        setTimeout(() => { apiStatusDiv.textContent = ''; }, 3000);
      });
    });
    
    // Load settings
    loadBotDetectionSettings();
  }, 500);
}

// Function to load bot detection settings
function loadBotDetectionSettings() {
  chrome.runtime.sendMessage({
    action: 'getBotDetectionSettings'
  }, (response) => {
    if (!response) return;
    
    const enabledCheckbox = document.getElementById('yt-bot-detection-enabled');
    const apiKeyInput = document.getElementById('yt-api-key');
    const apiUrlInput = document.getElementById('yt-bot-api-url');
    const markNamesCheckbox = document.getElementById('yt-bot-mark-names');
    const enableReportMenuCheckbox = document.getElementById('yt-enable-report-menu');
    
    if (enabledCheckbox) {
      enabledCheckbox.checked = response.enabled;
    }
    
    if (apiKeyInput && response.apiKey) {
      apiKeyInput.value = response.apiKey;
    }
    
    if (apiUrlInput && response.serverUrl) {
      apiUrlInput.value = response.serverUrl;
    }
    
    if (markNamesCheckbox) {
      markNamesCheckbox.checked = response.markBotNames !== false;
    }
    
    if (enableReportMenuCheckbox) {
      enableReportMenuCheckbox.checked = response.enableReportMenu !== false;
    }
  });
}

// Function to make an element draggable
function makeDraggable(element, dragHandle) {
  let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
  
  dragHandle.style.cursor = 'move';
  dragHandle.onmousedown = dragMouseDown;
  
  function dragMouseDown(e) {
    e.preventDefault();
    pos3 = e.clientX;
    pos4 = e.clientY;
    document.onmouseup = closeDragElement;
    document.onmousemove = elementDrag;
  }
  
  function elementDrag(e) {
    e.preventDefault();
    pos1 = pos3 - e.clientX;
    pos2 = pos4 - e.clientY;
    pos3 = e.clientX;
    pos4 = e.clientY;
    
    // Calculate new position while keeping the element within the viewport
    let newTop = element.offsetTop - pos2;
    let newLeft = element.offsetLeft - pos1;
    
    // Ensure the element stays within the viewport
    newTop = Math.max(0, Math.min(newTop, window.innerHeight - dragHandle.offsetHeight));
    newLeft = Math.max(0, Math.min(newLeft, window.innerWidth - element.offsetWidth));
    
    element.style.top = newTop + "px";
    element.style.left = newLeft + "px";
  }
  
  function closeDragElement() {
    document.onmouseup = null;
    document.onmousemove = null;
  }
}

// Function to intercept and modify YouTube API responses
function interceptYouTubeResponses() {
  console.log("Setting up YouTube API response interception");
  
  // Store the original fetch function
  const originalFetch = window.fetch;
  
  // Override the fetch function
  window.fetch = async function(resource, init) {
    try {
      // Check if this is a YouTube API request we want to modify
      const url = typeof resource === 'string' ? resource : resource.url;
      const method = init?.method || 'GET';
      const isYouTubeApiRequest = url && (url.includes('/youtubei/v1/next') || url.includes('/youtubei/v1/browse'));
      
      // For YouTube API requests, we want to potentially modify the response
      if (isYouTubeApiRequest) {
        console.log(`Intercepted YouTube API request: ${url}`);
        
        // Call the original fetch function
        const response = await originalFetch.apply(this, arguments);
        
        // Check if bot detection is enabled
        chrome.storage.local.get('botDetectionEnabled', async (data) => {
          if (!data.botDetectionEnabled) {
            console.log("Bot detection is disabled, skipping comment processing");
            return;
          }
          
          try {
            // Clone the response so we can read the body
            const clonedResponse = response.clone();
            
            // Parse the response as JSON
            const responseData = await clonedResponse.json();
            
            // Check if we have bot comments in the response
            const comments = extractCommentsFromResponse(responseData);
            if (comments && comments.length > 0) {
              console.log(`Found ${comments.length} comments in response`);
              
              // Send the comments to the background script for bot detection
              chrome.runtime.sendMessage({
                action: 'detectBots',
                comments: comments
              }, (result) => {
                if (chrome.runtime.lastError) {
                  console.error("Runtime error:", chrome.runtime.lastError);
                  return;
                }
                
                if (result && result.success && result.results) {
                  console.log('Bot detection results:', result.results);
                  
                  // Store the results for DOM modification
                  chrome.storage.local.set({ 
                    'lastBotDetectionResults': result.results,
                    'lastDetectionTime': Date.now()
                  });
                } else {
                  const errorMsg = result && result.error ? result.error : 'Unknown error';
                  console.error('Failed to detect bots:', errorMsg);
                }
              });
            } else {
              console.log("No comments found in the response");
            }
          } catch (error) {
            console.error('Error processing response:', error);
          }
        });
        
        // Return the original response
        return response;
      }
      
      // For all other requests, just pass through
      return originalFetch.apply(this, arguments);
    } catch (error) {
      console.error('Error in fetch interceptor:', error);
      // Ensure the original fetch is called even if our code fails
      return originalFetch.apply(this, arguments);
    }
  };
  
  console.log("YouTube API response interception set up successfully");
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

// Wait for page to load before adding panel
window.addEventListener('load', () => {
  // Check if we're on YouTube
  if (window.location.hostname.includes('youtube.com')) {
    // Add control panel
    addControlPanel();
    
    // Set up comment report menu
    setupCommentReportMenu();
    
    // Intercept YouTube API responses
    interceptYouTubeResponses();
    
    // Apply DOM modifications for bot comments
    setInterval(applyBotDetectionToDom, 2000);
  }
});

// Function to apply bot detection to the DOM
function applyBotDetectionToDom() {
  // Get settings from storage
  chrome.storage.local.get(['botDetectionEnabled', 'markBotNames', 'lastBotDetectionResults', 'lastDetectionTime'], (data) => {
    if (!data.botDetectionEnabled) return;
    
    // Check if we have detection results that are recent (less than 30 seconds old)
    const now = Date.now();
    const isRecentDetection = data.lastDetectionTime && (now - data.lastDetectionTime < 30000);
    
    if (isRecentDetection && data.lastBotDetectionResults) {
      console.log("Using recent bot detection results:", data.lastBotDetectionResults);
    }
    
    // Find all comment author elements that haven't been processed yet
    const commentAuthors = document.querySelectorAll('#author-text:not([data-bot-processed])');
    console.log(`Found ${commentAuthors.length} unprocessed comment authors`);
    
    // Process each comment author
    commentAuthors.forEach(author => {
      // Mark as processed
      author.setAttribute('data-bot-processed', 'true');
      
      // Get the author name
      const authorName = author.textContent.trim();
      
      // Check if the author name contains the [BOT] tag
      if (authorName.includes('[BOT]')) {
        // Add a visual indicator
        author.style.color = '#e74c3c';
        
        // Add a bot icon if mark names is enabled
        if (data.markBotNames !== false) {
          const botIcon = document.createElement('span');
          botIcon.textContent = '🤖';
          botIcon.style.marginLeft = '5px';
          author.appendChild(botIcon);
        }
      } 
      // If we have recent detection results, check if this author is in it
      else if (isRecentDetection && data.lastBotDetectionResults) {
        try {
          // Find the comment content associated with this author
          const commentElement = author.closest('ytd-comment-renderer');
          if (!commentElement) return;
          
          const contentElement = commentElement.querySelector('#content-text');
          if (!contentElement) return;
          
          const commentContent = contentElement.textContent.trim();
          
          console.log(`Checking comment: ${authorName} - "${commentContent.substring(0, 30)}..."`);
          
          // Check the format of the results
          if (data.lastBotDetectionResults.is_bot) {
            // Original format with is_bot array
            const index = data.lastBotDetectionResults.is_bot.findIndex((isBot, idx) => {
              // Try to match by original comments array if available
              if (data.lastBotDetectionResults.comments) {
                const comment = data.lastBotDetectionResults.comments[idx];
                return comment && 
                      (comment.author === authorName || comment.author_name === authorName) && 
                      (comment.content === commentContent);
              }
              
              // Otherwise, just use the index (less reliable)
              return false;
            });
            
            if (index !== -1 && data.lastBotDetectionResults.is_bot[index]) {
              console.log(`Marking comment as bot: ${authorName}`);
              markCommentAsBot(author, data.markBotNames);
            }
          } 
          // New format with results array
          else if (data.lastBotDetectionResults.results) {
            const index = data.lastBotDetectionResults.results.findIndex((result, idx) => {
              // Try to match by evaluate array if available
              if (data.lastBotDetectionResults.evaluate) {
                const comment = data.lastBotDetectionResults.evaluate[idx];
                return comment && 
                      (comment.author === authorName || comment.author_name === authorName) && 
                      (comment.content === commentContent);
              }
              
              // Otherwise, just use the index (less reliable)
              return false;
            });
            
            if (index !== -1 && data.lastBotDetectionResults.results[index]) {
              console.log(`Marking comment as bot: ${authorName}`);
              markCommentAsBot(author, data.markBotNames);
            }
          }
        } catch (error) {
          console.error('Error processing detection results for DOM updates:', error);
        }
      }
    });
  });
}

// Helper function to mark a comment as a bot
function markCommentAsBot(authorElement, markBotNames) {
  // Add a visual indicator
  authorElement.style.color = '#e74c3c';
  
  // Add the [BOT] tag to the displayed name
  if (!authorElement.textContent.includes('[BOT]')) {
    authorElement.textContent = authorElement.textContent + ' [BOT]';
  }
  
  // Add a bot icon if mark names is enabled
  if (markBotNames !== false) {
    const botIcon = document.createElement('span');
    botIcon.textContent = '🤖';
    botIcon.style.marginLeft = '5px';
    authorElement.appendChild(botIcon);
  }
}

// Function to set up the comment report menu
function setupCommentReportMenu() {
  // Create the context menu
  const contextMenu = createCustomContextMenu();
  
  // Add event listener for right clicks on comments
  document.addEventListener('contextmenu', handleCommentRightClick);
  
  // Listen for messages from the background script
  chrome.runtime.onMessage.addListener((message) => {
    if (message.action === 'contextMenuReportComment') {
      // Get the comment element under the cursor
      const commentElement = document.elementFromPoint(
        contextMenu.lastX || 0, 
        contextMenu.lastY || 0
      );
      
      if (commentElement) {
        const comment = findCommentElement(commentElement);
        if (comment) {
          const commentData = extractCommentData(comment);
          if (commentData) {
            reportComment(commentData);
          }
        }
      }
    }
  });
}

// Function to create a custom context menu
function createCustomContextMenu() {
  const menu = document.createElement('div');
  menu.id = 'yt-bot-detector-context-menu';
  menu.style.cssText = `
    position: absolute;
    background-color: rgba(33, 33, 33, 0.95);
    border-radius: 4px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    z-index: 10000;
    padding: 5px 0;
    display: none;
    font-family: 'Roboto', Arial, sans-serif;
    font-size: 14px;
    color: #fff;
    min-width: 150px;
  `;
  
  const reportItem = document.createElement('div');
  reportItem.textContent = 'Report as Bot';
  reportItem.style.cssText = `
    padding: 8px 15px;
    cursor: pointer;
  `;
  reportItem.addEventListener('mouseover', () => {
    reportItem.style.backgroundColor = '#065fd4';
  });
  reportItem.addEventListener('mouseout', () => {
    reportItem.style.backgroundColor = 'transparent';
  });
  reportItem.addEventListener('click', () => {
    hideContextMenu();
    
    // Get the current comment data
    const currentCommentData = menu.commentData;
    if (currentCommentData) {
      reportComment(currentCommentData);
    }
  });
  
  menu.appendChild(reportItem);
  document.body.appendChild(menu);
  
  // Hide the menu when clicking elsewhere
  document.addEventListener('click', hideContextMenu);
  
  return menu;
}

// Function to handle right clicks on comments
function handleCommentRightClick(event) {
  chrome.storage.local.get(['botDetectionEnabled', 'enableReportMenu'], (data) => {
    if (!data.botDetectionEnabled || data.enableReportMenu === false) return;
    
    const commentElement = findCommentElement(event.target);
    if (commentElement) {
      // Get comment data
      const commentData = extractCommentData(commentElement);
      if (commentData) {
        // Show the context menu
        showContextMenu(event.clientX, event.clientY, commentData);
        event.preventDefault();
      }
    }
  });
}

// Function to find a comment element from a clicked element
function findCommentElement(element) {
  // Try to find the closest comment renderer
  let current = element;
  while (current && current !== document.body) {
    if (current.tagName && 
        current.tagName.toLowerCase() === 'ytd-comment-renderer') {
      return current;
    }
    current = current.parentElement;
  }
  return null;
}

// Function to extract comment data from a comment element
function extractCommentData(commentElement) {
  try {
    // Get the author name
    const authorElement = commentElement.querySelector('#author-text');
    if (!authorElement) return null;
    
    const authorName = authorElement.textContent.trim();
    
    // Get the comment content
    const contentElement = commentElement.querySelector('#content-text');
    if (!contentElement) return null;
    
    const content = contentElement.textContent.trim();
    
    // Return the comment data
    return {
      author_name: authorName,
      content: content
    };
  } catch (error) {
    console.error('Error extracting comment data:', error);
    return null;
  }
}

// Function to show the context menu
function showContextMenu(x, y, commentData) {
  const menu = document.getElementById('yt-bot-detector-context-menu');
  if (!menu) return;
  
  // Store the comment data
  menu.commentData = commentData;
  menu.lastX = x;
  menu.lastY = y;
  
  // Position the menu
  menu.style.left = `${x}px`;
  menu.style.top = `${y}px`;
  menu.style.display = 'block';
}

// Function to hide the context menu
function hideContextMenu() {
  const menu = document.getElementById('yt-bot-detector-context-menu');
  if (menu) {
    menu.style.display = 'none';
  }
}

// Function to report a comment as a bot
function reportComment(commentData) {
  chrome.storage.local.get(['serverUrl', 'apiKey'], (data) => {
    if (!data.serverUrl || !data.apiKey) {
      console.error('Server URL or API key not configured');
      return;
    }
    
    chrome.runtime.sendMessage({
      action: 'reportComment',
      serverUrl: data.serverUrl,
      apiKey: data.apiKey,
      comment: commentData
    }, (response) => {
      if (response && response.success) {
        console.log('Comment reported successfully');
        
        // Find the comment in the DOM and mark it as a bot
        const commentElements = document.querySelectorAll('ytd-comment-renderer');
        for (const commentElement of commentElements) {
          const authorElement = commentElement.querySelector('#author-text');
          const contentElement = commentElement.querySelector('#content-text');
          
          if (authorElement && contentElement) {
            const authorName = authorElement.textContent.trim();
            const content = contentElement.textContent.trim();
            
            if (authorName === commentData.author_name && content === commentData.content) {
              // Mark this comment as a bot
              if (!authorElement.textContent.includes('[BOT]')) {
                authorElement.textContent = authorElement.textContent + ' [BOT]';
              }
              
              authorElement.style.color = '#e74c3c';
              
              // Add a bot icon
              chrome.storage.local.get('markBotNames', (data) => {
                if (data.markBotNames !== false) {
                  const botIcon = document.createElement('span');
                  botIcon.textContent = '🤖';
                  botIcon.style.marginLeft = '5px';
                  authorElement.appendChild(botIcon);
                }
              });
              
              break;
            }
          }
        }
      } else {
        const errorMsg = response && response.error ? response.error : 'Unknown error';
        console.error('Error reporting comment:', errorMsg);
      }
    });
  });
} 