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
        if (chrome.runtime.lastError) {
          apiStatusDiv.textContent = `Error: ${chrome.runtime.lastError.message}`;
          apiStatusDiv.style.color = '#f44336';
          return;
        }
        
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
      
      if (enabled && !apiKey) {
        apiStatusDiv.textContent = 'Error: API Key is required';
        apiStatusDiv.style.color = '#f44336';
        return;
      }
      
      // Save settings to storage first
      chrome.storage.local.set({
        'botDetectionEnabled': enabled,
        'serverUrl': serverUrl,
        'apiKey': apiKey,
        'markBotNames': markBotNames,
        'enableReportMenu': enableReportMenu
      }, () => {
        if (chrome.runtime.lastError) {
          apiStatusDiv.textContent = `Error saving settings: ${chrome.runtime.lastError.message}`;
          apiStatusDiv.style.color = '#f44336';
          return;
        }
        
        // Then update the background script
        chrome.runtime.sendMessage({
          action: 'setBotDetectionSettings',
          enabled: enabled,
          serverUrl: serverUrl,
          apiKey: apiKey,
          markBotNames: markBotNames,
          enableReportMenu: enableReportMenu
        }, (response) => {
          if (chrome.runtime.lastError) {
            apiStatusDiv.textContent = `Error: ${chrome.runtime.lastError.message}`;
            apiStatusDiv.style.color = '#f44336';
            return;
          }
          
          if (response && response.success) {
            apiStatusDiv.textContent = enabled ? 'Bot detection enabled!' : 'Bot detection disabled!';
            apiStatusDiv.style.color = '#4caf50';
            
            // If enabled and properly configured, trigger an immediate check
            if (enabled && serverUrl && apiKey) {
              console.log("✅ Bot detection settings updated, triggering immediate check");
              setTimeout(() => applyBotDetectionToDom(), 1000);
            }
          } else {
            const errorMsg = response && response.error ? response.error : 'Unknown error';
            apiStatusDiv.textContent = `Error: ${errorMsg}`;
            apiStatusDiv.style.color = '#f44336';
          }
          setTimeout(() => { apiStatusDiv.textContent = ''; }, 3000);
        });
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
      
      // More specific matching for YouTube API requests that contain comments
      const isYouTubeApiRequest = url && (
        url.includes('/youtubei/v1/next') || 
        url.includes('/youtubei/v1/browse') ||
        // Add more specific patterns for comment-related requests
        url.includes('/youtubei/v1/comment/get_comment_replies') ||
        url.includes('/youtubei/v1/comment/create_comment') ||
        url.includes('/youtubei/v1/comment/get_comments')
      );
      
      // For YouTube API requests, we want to potentially modify the response
      if (isYouTubeApiRequest) {
        console.log(`🔍 Intercepted YouTube API request: ${url}`);
        console.log(`📊 Request method: ${method}, Body size: ${init && init.body ? init.body.length : 'N/A'}`);
        
        // Call the original fetch function
        const response = await originalFetch.apply(this, arguments);
        console.log(`✅ Got response for: ${url}, Status: ${response.status}`);
        
        // Check if bot detection is enabled
        try {
          chrome.storage.local.get(['botDetectionEnabled', 'serverUrl', 'apiKey'], async (data) => {
            try {
              if (chrome.runtime.lastError) {
                console.error("Runtime error in storage.local.get:", chrome.runtime.lastError);
                return;
              }
              
              console.log(`🔧 Bot detection settings: enabled=${data.botDetectionEnabled}, serverUrl=${data.serverUrl ? 'set' : 'not set'}, apiKey=${data.apiKey ? 'set' : 'not set'}`);
              
              if (!data.botDetectionEnabled) {
                console.log("⚠️ Bot detection is disabled, skipping comment processing");
                return;
              }
              
              if (!data.serverUrl) {
                console.error("⛔ Server URL not configured, cannot process comments");
                return;
              }
              
              if (!data.apiKey) {
                console.error("⛔ API key not configured, cannot process comments");
                return;
              }
              
              try {
                // Clone the response so we can read the body
                const clonedResponse = response.clone();
                
                // Parse the response as JSON
                const responseData = await clonedResponse.json();
                console.log(`📦 Response data received, type: ${typeof responseData}, has data: ${!!responseData}`);
                
                // Check if we have bot comments in the response
                const comments = extractCommentsFromResponse(responseData);
                if (comments && comments.length > 0) {
                  console.log(`📝 Found ${comments.length} comments in response:`);
                  // Log a sample of comments (first 2)
                  comments.slice(0, 2).forEach((comment, i) => {
                    console.log(`  Comment ${i+1}: Author="${comment.author_name}", Content="${comment.content.substring(0, 30)}..."`);
                  });
                  
                  // Send the comments to the background script for bot detection
                  try {
                    console.log(`🚀 Sending ${comments.length} comments to background script for detection`);
                    chrome.runtime.sendMessage({
                      action: 'detectBots',
                      comments: comments
                    }, (result) => {
                      try {
                        if (chrome.runtime.lastError) {
                          console.error("⚠️ Runtime error in sendMessage callback:", chrome.runtime.lastError);
                          return;
                        }
                        
                        console.log(`📨 Got response from background script:`, result);
                        
                        if (result && result.success && result.results) {
                          console.log('✅ Bot detection results received:', result.results);
                          
                          // Store the results for DOM modification
                          try {
                            chrome.storage.local.set({ 
                              'lastBotDetectionResults': result.results,
                              'lastDetectionTime': Date.now()
                            }, () => {
                              if (chrome.runtime.lastError) {
                                console.error("⚠️ Runtime error in storage.local.set:", chrome.runtime.lastError);
                              } else {
                                console.log("💾 Stored bot detection results for DOM updates");
                              }
                            });
                          } catch (storageError) {
                            console.error('⚠️ Error storing bot detection results:', storageError);
                          }
                        } else {
                          const errorMsg = result && result.error ? result.error : 'Unknown error';
                          console.error('❌ Failed to detect bots:', errorMsg);
                        }
                      } catch (callbackError) {
                        console.error('⚠️ Error in sendMessage callback:', callbackError);
                      }
                    });
                  } catch (sendMessageError) {
                    console.error('⚠️ Error sending message to background script:', sendMessageError);
                  }
                } else {
                  console.log("ℹ️ No comments found in the response");
                }
              } catch (responseError) {
                console.error('⚠️ Error processing response:', responseError);
              }
            } catch (storageCallbackError) {
              console.error('⚠️ Error in storage callback:', storageCallbackError);
            }
          });
        } catch (storageError) {
          console.error('⚠️ Error accessing chrome.storage:', storageError);
        }
        
        // Return the original response
        return response;
      }
      
      // For all other requests, just pass through
      return originalFetch.apply(this, arguments);
    } catch (error) {
      console.error('⚠️ Error in fetch interceptor:', error);
      // Ensure the original fetch is called even if our code fails
      return originalFetch.apply(this, arguments);
    }
  };
  
  console.log("✅ YouTube API response interception set up successfully");
}

// Function to extract comments from YouTube response
function extractCommentsFromResponse(responseData) {
  try {
    console.log("🔍 Analyzing response data for comments");
    
    if (!responseData) {
      console.log("⚠️ Response data is null or undefined");
      return null;
    }
    
    // Log the structure of the response data to help debug
    console.log("📊 Response data structure:", 
      Object.keys(responseData).join(", "),
      responseData.frameworkUpdates ? "has frameworkUpdates" : "no frameworkUpdates"
    );
    
    if (!responseData.frameworkUpdates) {
      console.log("⚠️ No frameworkUpdates in response");
      return null;
    }
    
    if (!responseData.frameworkUpdates.entityBatchUpdate) {
      console.log("⚠️ No entityBatchUpdate in frameworkUpdates");
      return null;
    }
    
    if (!responseData.frameworkUpdates.entityBatchUpdate.mutations) {
      console.log("⚠️ No mutations in entityBatchUpdate");
      return null;
    }
    
    const {mutations} = responseData.frameworkUpdates.entityBatchUpdate;
    console.log(`📋 Found ${mutations.length} mutations in response`);
    
    const comments = [];
    let commentPayloads = 0;
    
    for (const mutation of mutations) {
      if (mutation.payload && mutation.payload.commentEntityPayload) {
        commentPayloads++;
        const payload = mutation.payload.commentEntityPayload;
        
        if (payload.author && payload.author.displayName && 
            payload.properties && payload.properties.content) {
          comments.push({
            author_name: payload.author.displayName,
            content: payload.properties.content.content
          });
        }
      }
    }
    
    console.log(`📝 Found ${commentPayloads} comment payloads, extracted ${comments.length} valid comments`);
    
    return comments.length > 0 ? comments : null;
  } catch (error) {
    console.error('⚠️ Error extracting comments:', error);
    return null;
  }
}

// Wait for page to load before adding panel
window.addEventListener('load', () => {
  try {
    // Check if we're on YouTube
    if (window.location.hostname.includes('youtube.com')) {
      console.log("YouTube detected, initializing bot detector");
      
      try {
        // Add control panel
        addControlPanel();
        
        
        
        // Intercept YouTube API responses
        interceptYouTubeResponses();
        
        // Load settings and verify configuration
        chrome.storage.local.get(['botDetectionEnabled', 'serverUrl', 'apiKey'], (data) => {
          try {
            if (chrome.runtime.lastError) {
              console.error("Runtime error loading settings:", chrome.runtime.lastError);
              return;
            }
            
            console.log(`🔧 Bot detector settings loaded: enabled=${data.botDetectionEnabled}, serverUrl=${data.serverUrl ? 'set' : 'not set'}, apiKey=${data.apiKey ? 'set' : 'not set'}`);
            
            if (!data.botDetectionEnabled) {
              console.log("⚠️ Bot detection is currently disabled");
            } else if (!data.serverUrl) {
              console.error("⛔ Server URL not configured, bot detection will not work");
            } else if (!data.apiKey) {
              console.error("⛔ API key not configured, bot detection will not work");
            } else {
              console.log("✅ Bot detection is properly configured");
              
              // Trigger an immediate check for comments
              setTimeout(() => {
                console.log("🔍 Triggering initial comment check");
                applyBotDetectionToDom();
              }, 3000);
            }
          } catch (settingsError) {
            console.error("Error processing settings:", settingsError);
          }
        });
        
        // Set up interval for DOM modifications
        const intervalId = setInterval(() => {
          try {
            applyBotDetectionToDom();
          } catch (intervalError) {
            console.error("Error in applyBotDetectionToDom interval:", intervalError);
            
            // If we get an extension context invalidated error, clear the interval
            if (intervalError.message && intervalError.message.includes("Extension context invalidated")) {
              console.log("Extension context invalidated, clearing interval");
              clearInterval(intervalId);
            }
          }
        }, 5000); // Changed to 5 seconds to reduce API calls
        
        // Clean up on unload
        window.addEventListener('unload', () => {
          clearInterval(intervalId);
        });
      } catch (initError) {
        console.error("Error initializing bot detector:", initError);
      }
    }
  } catch (error) {
    console.error("Error in load event listener:", error);
  }
});

// Function to apply bot detection to the DOM
function applyBotDetectionToDom() {
  try {
    // Get settings from storage
    chrome.storage.local.get(['botDetectionEnabled', 'markBotNames', 'lastBotDetectionResults', 'lastDetectionTime', 'serverUrl', 'apiKey'], (data) => {
      try {
        if (chrome.runtime.lastError) {
          console.error("Runtime error in storage.local.get:", chrome.runtime.lastError);
          return;
        }
        
        if (!data.botDetectionEnabled) return;
        
        // Find all comment author elements that haven't been processed yet
        const commentAuthors = document.querySelectorAll('#author-text:not([data-bot-processed])');
        if (commentAuthors.length > 0) {
            console.log(`Found ${commentAuthors.length} unprocessed comment authors`);
        }
        
        if (commentAuthors.length === 0) return;
        
        // Check if we have necessary settings for API call
        if (!data.serverUrl) {
          console.error("⛔ Server URL not configured, cannot detect bots");
          return;
        }
        
        if (!data.apiKey) {
          console.error("⛔ API key not configured, cannot detect bots");
          return;
        }
        
        // Collect comments to analyze
        const commentsToAnalyze = [];
        
        // Process each comment author
        commentAuthors.forEach((author, index) => {
          try {
            // Mark as processed to avoid processing again
            author.setAttribute('data-bot-processed', 'true');
            
            // Get the author name
            const authorName = author.textContent.trim();
            
            // Check if this comment is already marked as a bot
            if (author.hasAttribute('data-is-bot')) {
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
            // Check if this comment is already marked as a user
            else if (author.hasAttribute('data-is-user')) {
              // Make sure it has the user checkmark
              let hasCheckmark = false;
              const spans = author.querySelectorAll('span');
              spans.forEach(span => {
                if (span.textContent === '✓') {
                  hasCheckmark = true;
                }
              });
              
              if (!hasCheckmark) {
                const userIcon = document.createElement('span');
                userIcon.textContent = '✓';
                userIcon.style.cssText = `
                  margin-left: 5px;
                  color: #2ecc71;
                  font-weight: bold;
                `;
                author.appendChild(userIcon);
              }
            }
            else {
              // Find the comment content
              const commentElement = author.closest('ytd-comment-view-model');
              if (!commentElement) {
                console.log(`[Debug] Could not find 'ytd-comment-view-model' for author #${index} ('${authorName}')`);
                return;
              }
              
              const contentElement = commentElement.querySelector('#content-text');
              if (!contentElement) {
                console.log(`[Debug] Could not find '#content-text' in comment for author #${index} ('${authorName}')`);
                return;
              }
              
              const commentContent = contentElement.textContent.trim();
              
              // Add to comments to analyze
              commentsToAnalyze.push({
                author_name: authorName,
                content: commentContent,
                element: author // Store reference to the DOM element
              });
            }
          } catch (commentError) {
            console.error('Error processing comment:', commentError);
          }
        });
        
        // If we have comments to analyze, send them to the API
        if (commentsToAnalyze.length > 0) {
          console.log(`🚀 Sending ${commentsToAnalyze.length} comments directly to bot detection API`);
          
          // Send directly to the background script
          chrome.runtime.sendMessage({
            action: 'detectBots',
            comments: commentsToAnalyze.map(c => ({ author_name: c.author_name, content: c.content }))
          }, (result) => {
            try {
              if (chrome.runtime.lastError) {
                console.error("⚠️ Runtime error in sendMessage callback:", chrome.runtime.lastError);
                return;
              }
              
              console.log(`📨 Got response from background script:`, result);
              
              if (result && result.success && result.results) {
                console.log('✅ Bot detection results received:', result.results);
                
                // Apply the results to the DOM
                if (result.results.is_bot && Array.isArray(result.results.is_bot)) {
                  result.results.is_bot.forEach((isBot, index) => {
                    if (isBot && index < commentsToAnalyze.length) {
                      const authorElement = commentsToAnalyze[index].element;
                      console.log(`Marking comment as bot: ${commentsToAnalyze[index].author_name}`);
                      markCommentAsBot(authorElement, data.markBotNames);
                    }
                  });
                }
                
                // Store the results for future use
                try {
                  chrome.storage.local.set({ 
                    'lastBotDetectionResults': result.results,
                    'lastDetectionTime': Date.now()
                  }, () => {
                    if (chrome.runtime.lastError) {
                      console.error("⚠️ Runtime error in storage.local.set:", chrome.runtime.lastError);
                    } else {
                      console.log("💾 Stored bot detection results");
                    }
                  });
                } catch (storageError) {
                  console.error('⚠️ Error storing bot detection results:', storageError);
                }
              } else {
                const errorMsg = result && result.error ? result.error : 'Unknown error';
                console.error('❌ Failed to detect bots:', errorMsg);
              }
            } catch (callbackError) {
              console.error('⚠️ Error in sendMessage callback:', callbackError);
            }
          });
        }
      } catch (error) {
        console.error('Error in applyBotDetectionToDom storage callback:', error);
      }
    });
  } catch (error) {
    console.error('Error in applyBotDetectionToDom:', error);
  }
}

// Helper function to mark a comment as a bot
function markCommentAsBot(authorElement, markBotNames) {
  // Add a visual indicator
  authorElement.style.color = '#e74c3c';
  
  // Set data attribute to mark as bot
  authorElement.setAttribute('data-is-bot', 'true');
  
  // Add a bot icon if mark names is enabled
  if (markBotNames !== false) {
    const botIcon = document.createElement('span');
    botIcon.textContent = '🤖';
    botIcon.style.marginLeft = '5px';
    authorElement.appendChild(botIcon);
  }
}

// Helper function to mark a comment as a user
function markCommentAsUser(authorElement) {
  // Remove bot marking if present
  if (authorElement.hasAttribute('data-is-bot')) {
    authorElement.removeAttribute('data-is-bot');
  }
  
  // Reset color to default
  authorElement.style.color = '';
  
  // Set data attribute to mark as user-verified
  authorElement.setAttribute('data-is-user', 'true');
  
  // Remove bot icon if present
  const existingIcons = authorElement.querySelectorAll('span');
  existingIcons.forEach(icon => {
    if (icon.textContent === '🤖') {
      authorElement.removeChild(icon);
    }
  });
  
  // Add a visual indicator for human-verified comments
  // First check if it already has the checkmark
  let hasCheckmark = false;
  existingIcons.forEach(icon => {
    if (icon.textContent === '✓') {
      hasCheckmark = true;
    }
  });
  
  if (!hasCheckmark) {
    const userIcon = document.createElement('span');
    userIcon.textContent = '✓';
    userIcon.style.cssText = `
      margin-left: 5px;
      color: #2ecc71;
      font-weight: bold;
    `;
    authorElement.appendChild(userIcon);
  }
}

// Helper function to find the comment element from a given target
function findCommentElement(target) {
  let element = target;
  while (element && !element.matches('ytd-comment-view-model')) {
    element = element.parentElement;
  }
  if (element) {
    console.log('Found comment element:', element);
  } else {
    console.log('No comment element found for target:', target);
  }
  return element;
}

// Helper function to extract comment data from a comment element
function extractCommentData(commentElement) {
  const authorElement = commentElement.querySelector('#author-text');
  const contentElement = commentElement.querySelector('#content-text');

  if (authorElement && contentElement) {
    return {
      author_name: authorElement.textContent.trim(),
      content: contentElement.textContent.trim()
    };
  }
  return null;
}

let lastClickedCommentData = null;

document.addEventListener('contextmenu', (event) => {
  const commentElement = findCommentElement(event.target);
  if (commentElement) {
    lastClickedCommentData = extractCommentData(commentElement);
    lastClickedCommentData.author_name = lastClickedCommentData.author_name.replace('🤖', '').trim()
    if (lastClickedCommentData) {
      console.log('Comment data extracted for context menu:', lastClickedCommentData);
    } else {
      console.warn('Could not extract comment data from found comment element.');
    }
  } else {
    lastClickedCommentData = null; // Ensure it's null if no comment element is found
    console.log('Context menu opened, but no comment element was clicked.');
  }
});

// Listen for messages from the background script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'contextMenuClick') {
    if (lastClickedCommentData) {
      const isBot = message.menuItemId === 'reportAsBot';
      reportComment(lastClickedCommentData, isBot);
      lastClickedCommentData = null; // Clear the stored data after use
    } else {
      console.warn('No comment data available for reporting.');
    }
  }
});;


    

  

// Function to report a comment as a bot or user
function reportComment(commentData, isBot) {
  try {
    chrome.storage.local.get(['serverUrl', 'apiKey'], (data) => {
      try {
        if (chrome.runtime.lastError) {
          console.error("Runtime error in storage.local.get:", chrome.runtime.lastError);
          return;
        }
        
        if (!data.serverUrl || !data.apiKey) {
          console.error('Server URL or API key not configured');
          return;
        }
        
        try {
          console.log(`Reporting comment as ${isBot ? 'bot' : 'user'}: "${commentData.content.substring(0, 30)}..."`);
          
          chrome.runtime.sendMessage({
            action: 'reportComment',
            serverUrl: data.serverUrl,
            apiKey: data.apiKey,
            comment: commentData,
            isBot: isBot
          }, (response) => {
            try {
              if (chrome.runtime.lastError) {
                console.error("Runtime error in sendMessage callback:", chrome.runtime.lastError);
                return;
              }
              
              if (response && response.success) {
                console.log(`Comment reported successfully as ${isBot ? 'bot' : 'user'}`);
                
                // Find the comment in the DOM and update its status
                const commentElements = document.querySelectorAll('ytd-comment-renderer');
                for (const commentElement of commentElements) {
                  const authorElement = commentElement.querySelector('#author-text');
                  const contentElement = commentElement.querySelector('#content-text');
                  
                  if (authorElement && contentElement) {
                    const authorName = authorElement.textContent.trim();
                    const content = contentElement.textContent.trim();
                    
                    if (authorName === commentData.author_name && content === commentData.content) {
                      if (isBot) {
                        // Mark this comment as a bot using the data attribute
                        markCommentAsBot(authorElement, true);
                      } else {
                        // Mark this comment as a user
                        markCommentAsUser(authorElement);
                      }
                      break;
                    }
                  }
                }
              } else {
                const errorMsg = response && response.error ? response.error : 'Unknown error';
                console.error('Error reporting comment:', errorMsg);
              }
            } catch (callbackError) {
              console.error('Error in sendMessage callback:', callbackError);
            }
          });
        } catch (sendMessageError) {
          console.error('Error sending message to background script:', sendMessageError);
        }
      } catch (storageCallbackError) {
        console.error('Error in storage callback:', storageCallbackError);
      }
    });
  } catch (error) {
    console.error('Error in reportComment:', error);
  }
} 