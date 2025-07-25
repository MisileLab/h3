// Content script for YouTube Request Interceptor

// Function to add a control panel to the YouTube page
function addControlPanel() {
  // Create control panel container
  const panel = document.createElement('div');
  panel.id = 'yt-request-interceptor-panel';
  panel.style.cssText = `
    position: fixed;
    bottom: 20px;
    right: 20px;
    background: rgba(33, 33, 33, 0.9);
    color: white;
    padding: 10px;
    border-radius: 5px;
    z-index: 9999;
    font-family: Arial, sans-serif;
    max-width: 300px;
    max-height: 400px;
    overflow-y: auto;
    box-shadow: 0 0 10px rgba(0,0,0,0.5);
  `;

  // Add header
  const header = document.createElement('div');
  header.innerHTML = '<strong>YouTube Request Interceptor</strong>';
  header.style.cssText = `
    margin-bottom: 10px;
    padding-bottom: 5px;
    border-bottom: 1px solid #555;
  `;
  panel.appendChild(header);

  // Add toggle button
  const toggleBtn = document.createElement('button');
  toggleBtn.textContent = 'Show Requests';
  toggleBtn.style.cssText = `
    background: #065fd4;
    color: white;
    border: none;
    padding: 5px 10px;
    border-radius: 3px;
    cursor: pointer;
    margin-right: 5px;
  `;
  
  // Add response modification button
  const modifyBtn = document.createElement('button');
  modifyBtn.textContent = 'Modify Responses';
  modifyBtn.style.cssText = `
    background: #555;
    color: white;
    border: none;
    padding: 5px 10px;
    border-radius: 3px;
    cursor: pointer;
    margin-right: 5px;
  `;
  
  // Add bot detection button
  const botDetectionBtn = document.createElement('button');
  botDetectionBtn.textContent = 'Bot Detection';
  botDetectionBtn.style.cssText = `
    background: #555;
    color: white;
    border: none;
    padding: 5px 10px;
    border-radius: 3px;
    cursor: pointer;
  `;
  
  // Add requests container
  const requestsContainer = document.createElement('div');
  requestsContainer.id = 'yt-intercepted-requests';
  requestsContainer.style.cssText = `
    margin-top: 10px;
    display: none;
    font-size: 12px;
  `;
  
  // Add response modification container
  const modifyContainer = document.createElement('div');
  modifyContainer.id = 'yt-response-modification';
  modifyContainer.style.cssText = `
    margin-top: 10px;
    display: none;
    font-size: 12px;
  `;
  
  // Add bot detection container
  const botDetectionContainer = document.createElement('div');
  botDetectionContainer.id = 'yt-bot-detection';
  botDetectionContainer.style.cssText = `
    margin-top: 10px;
    display: none;
    font-size: 12px;
  `;
  
  // Create response modification UI
  modifyContainer.innerHTML = `
    <div style="margin-bottom: 10px;">
      <label style="display: block; margin-bottom: 5px;">
        <input type="checkbox" id="yt-modify-enabled" /> Enable Response Modification
      </label>
    </div>
    <div style="margin-bottom: 10px;">
      <div style="margin-bottom: 5px;">Example Server URL:</div>
      <div style="display: flex;">
        <input type="text" id="yt-server-url" style="flex-grow: 1; background: #222; color: #eee; border: 1px solid #444; padding: 5px; font-family: monospace; border-radius: 3px;" placeholder="https://example.com/api" />
        <button id="yt-test-server" style="margin-left: 5px; background: #555; color: white; border: none; padding: 5px; border-radius: 3px; cursor: pointer;">Test</button>
      </div>
      <div id="yt-server-status" style="margin-top: 5px; font-size: 11px;"></div>
    </div>
    <div style="margin-bottom: 10px;">
      <div style="margin-bottom: 5px;">API Key:</div>
      <div style="display: flex;">
        <input type="password" id="yt-modify-api-key" style="flex-grow: 1; background: #222; color: #eee; border: 1px solid #444; padding: 5px; font-family: monospace; border-radius: 3px;" placeholder="Your API key" />
        <button id="yt-toggle-modify-api-key" style="margin-left: 5px; background: #555; color: white; border: none; padding: 5px; border-radius: 3px; cursor: pointer;">Show</button>
      </div>
    </div>
    <div style="margin-bottom: 10px;">
      <div style="margin-bottom: 5px;">Response Modification Function:</div>
      <textarea id="yt-modify-function" style="width: 100%; height: 100px; background: #222; color: #eee; border: 1px solid #444; padding: 5px; font-family: monospace; resize: vertical; border-radius: 3px;" placeholder="// Write a function that takes a response object and URL
// Example:
function(response, url) {
  // Modify response here based on URL
  if (url.includes('/youtubei/v1/browse')) {
    response.data = 'Modified data';
  }
  return response;
}"></textarea>
    </div>
    <div>
      <button id="yt-modify-save" style="background: #065fd4; color: white; border: none; padding: 5px 10px; border-radius: 3px; cursor: pointer;">Save Function</button>
      <span id="yt-modify-status" style="margin-left: 10px; font-size: 12px;"></span>
    </div>
  `;
  
  // Create bot detection UI
  botDetectionContainer.innerHTML = `
    <div style="margin-bottom: 10px;">
      <label style="display: block; margin-bottom: 5px;">
        <input type="checkbox" id="yt-bot-detection-enabled" /> Enable Bot Detection
      </label>
    </div>
    <div style="margin-bottom: 10px;">
      <div style="margin-bottom: 5px;">API Key:</div>
      <div style="display: flex;">
        <input type="password" id="yt-api-key" style="flex-grow: 1; background: #222; color: #eee; border: 1px solid #444; padding: 5px; font-family: monospace; border-radius: 3px;" placeholder="Your API key" />
        <button id="yt-toggle-api-key" style="margin-left: 5px; background: #555; color: white; border: none; padding: 5px; border-radius: 3px; cursor: pointer;">Show</button>
      </div>
    </div>
    <div style="margin-bottom: 10px;">
      <div style="margin-bottom: 5px;">API Server URL:</div>
      <input type="text" id="yt-bot-api-url" style="width: 100%; background: #222; color: #eee; border: 1px solid #444; padding: 5px; font-family: monospace; border-radius: 3px;" placeholder="https://example.com/api" />
    </div>
    <div style="margin-bottom: 10px;">
      <button id="yt-test-bot-api" style="background: #555; color: white; border: none; padding: 5px 10px; border-radius: 3px; cursor: pointer;">Test API Connection</button>
      <div id="yt-bot-api-status" style="margin-top: 5px; font-size: 11px;"></div>
    </div>
    <div style="margin-bottom: 10px;">
      <div style="margin-bottom: 5px;">Bot Detection Settings:</div>
      <div style="margin-bottom: 5px;">
        <label style="display: block; margin-bottom: 5px;">
          <input type="checkbox" id="yt-bot-mark-names" checked /> Mark bot names with [BOT]
        </label>
      </div>
      <div style="margin-bottom: 5px;">
        <label style="display: block; margin-bottom: 5px;">
          <input type="checkbox" id="yt-enable-report-menu" checked /> Enable right-click report menu
        </label>
      </div>
    </div>
    <div>
      <button id="yt-bot-detection-save" style="background: #065fd4; color: white; border: none; padding: 5px 10px; border-radius: 3px; cursor: pointer;">Save Settings</button>
      <span id="yt-bot-detection-status" style="margin-left: 10px; font-size: 12px;"></span>
    </div>
  `;
  
  // Toggle display of requests
  toggleBtn.addEventListener('click', () => {
    if (requestsContainer.style.display === 'none') {
      requestsContainer.style.display = 'block';
      modifyContainer.style.display = 'none';
      botDetectionContainer.style.display = 'none';
      toggleBtn.textContent = 'Hide Requests';
      modifyBtn.style.background = '#555';
      botDetectionBtn.style.background = '#555';
      toggleBtn.style.background = '#065fd4';
      updateRequestsList();
    } else {
      requestsContainer.style.display = 'none';
      toggleBtn.textContent = 'Show Requests';
    }
  });
  
  // Toggle display of response modification
  modifyBtn.addEventListener('click', () => {
    if (modifyContainer.style.display === 'none') {
      modifyContainer.style.display = 'block';
      requestsContainer.style.display = 'none';
      botDetectionContainer.style.display = 'none';
      modifyBtn.style.background = '#065fd4';
      toggleBtn.style.background = '#555';
      botDetectionBtn.style.background = '#555';
      toggleBtn.textContent = 'Show Requests';
      loadResponseModificationSettings();
    } else {
      modifyContainer.style.display = 'none';
      modifyBtn.style.background = '#555';
    }
  });
  
  // Toggle display of bot detection
  botDetectionBtn.addEventListener('click', () => {
    if (botDetectionContainer.style.display === 'none') {
      botDetectionContainer.style.display = 'block';
      requestsContainer.style.display = 'none';
      modifyContainer.style.display = 'none';
      botDetectionBtn.style.background = '#065fd4';
      toggleBtn.style.background = '#555';
      modifyBtn.style.background = '#555';
      toggleBtn.textContent = 'Show Requests';
      loadBotDetectionSettings();
    } else {
      botDetectionContainer.style.display = 'none';
      botDetectionBtn.style.background = '#555';
    }
  });
  
  header.appendChild(toggleBtn);
  header.appendChild(modifyBtn);
  header.appendChild(botDetectionBtn);
  panel.appendChild(requestsContainer);
  panel.appendChild(modifyContainer);
  panel.appendChild(botDetectionContainer);
  
  // Add panel to page
  document.body.appendChild(panel);
  
  // Make panel draggable
  makeDraggable(panel, header);
  
  // Set up response modification events
  setupResponseModification();
  
  // Set up bot detection events
  setupBotDetection();
  
  return { panel, requestsContainer, modifyContainer, botDetectionContainer };
}

// Function to set up response modification events
function setupResponseModification() {
  // Wait for elements to be available
  setTimeout(() => {
    const enabledCheckbox = document.getElementById('yt-modify-enabled');
    const functionTextarea = document.getElementById('yt-modify-function');
    const serverUrlInput = document.getElementById('yt-server-url');
    const testServerBtn = document.getElementById('yt-test-server');
    const serverStatusDiv = document.getElementById('yt-server-status');
    const apiKeyInput = document.getElementById('yt-modify-api-key');
    const toggleApiKeyBtn = document.getElementById('yt-toggle-modify-api-key');
    const saveButton = document.getElementById('yt-modify-save');
    const statusSpan = document.getElementById('yt-modify-status');
    
    if (!enabledCheckbox || !functionTextarea || !saveButton || !statusSpan || 
        !serverUrlInput || !testServerBtn || !apiKeyInput || !toggleApiKeyBtn) return;
    
    // Toggle API key visibility
    toggleApiKeyBtn.addEventListener('click', () => {
      if (apiKeyInput.type === 'password') {
        apiKeyInput.type = 'text';
        toggleApiKeyBtn.textContent = 'Hide';
      } else {
        apiKeyInput.type = 'password';
        toggleApiKeyBtn.textContent = 'Show';
      }
    });
    
    // Test server button click handler
    testServerBtn.addEventListener('click', () => {
      const serverUrl = serverUrlInput.value.trim();
      const apiKey = apiKeyInput.value.trim();
      
      if (!serverUrl) {
        serverStatusDiv.textContent = 'Please enter a server URL';
        serverStatusDiv.style.color = '#ff9800';
        return;
      }
      
      serverStatusDiv.textContent = 'Testing connection...';
      serverStatusDiv.style.color = '#ff9800';
      
      chrome.runtime.sendMessage({
        action: 'testServerConnection',
        serverUrl: serverUrl,
        apiKey: apiKey
      }, (response) => {
        if (response.success) {
          serverStatusDiv.textContent = `Connection successful (${response.status})`;
          serverStatusDiv.style.color = '#4caf50';
        } else {
          serverStatusDiv.textContent = `Connection failed: ${response.error}`;
          serverStatusDiv.style.color = '#f44336';
        }
      });
    });
    
    // Save button click handler
    saveButton.addEventListener('click', () => {
      const enabled = enabledCheckbox.checked;
      const functionCode = functionTextarea.value.trim();
      const serverUrl = serverUrlInput.value.trim();
      const apiKey = apiKeyInput.value.trim();
      
      // Validate function code
      if (enabled && !functionCode) {
        statusSpan.textContent = 'Error: Function code is empty';
        statusSpan.style.color = '#f44336';
        return;
      }
      
      // Send to background script
      chrome.runtime.sendMessage({
        action: 'setResponseModification',
        enabled: enabled,
        functionCode: functionCode,
        serverUrl: serverUrl,
        apiKey: apiKey
      }, (response) => {
        if (response.success) {
          statusSpan.textContent = enabled ? 'Enabled!' : 'Disabled!';
          statusSpan.style.color = '#4caf50';
          
          // Save to storage
          chrome.storage.local.set({
            'responseModificationEnabled': enabled,
            'responseModificationFunction': functionCode,
            'exampleServerUrl': serverUrl,
            'modifyApiKey': apiKey
          });
        } else {
          statusSpan.textContent = `Error: ${response.error}`;
          statusSpan.style.color = '#f44336';
        }
        
        // Clear status after 3 seconds
        setTimeout(() => {
          statusSpan.textContent = '';
        }, 3000);
      });
    });
  }, 500);
}

// Function to set up bot detection events
function setupBotDetection() {
  // Wait for elements to be available
  setTimeout(() => {
    const enabledCheckbox = document.getElementById('yt-bot-detection-enabled');
    const apiKeyInput = document.getElementById('yt-api-key');
    const toggleApiKeyBtn = document.getElementById('yt-toggle-api-key');
    const apiUrlInput = document.getElementById('yt-bot-api-url');
    const testApiBtn = document.getElementById('yt-test-bot-api');
    const apiStatusDiv = document.getElementById('yt-bot-api-status');
    const markNamesCheckbox = document.getElementById('yt-bot-mark-names');
    const enableReportMenuCheckbox = document.getElementById('yt-enable-report-menu');
    const saveButton = document.getElementById('yt-bot-detection-save');
    const statusSpan = document.getElementById('yt-bot-detection-status');
    
    if (!enabledCheckbox || !apiKeyInput || !toggleApiKeyBtn || !apiUrlInput || 
        !testApiBtn || !apiStatusDiv || !markNamesCheckbox || !enableReportMenuCheckbox || !saveButton || !statusSpan) return;
    
    // Toggle API key visibility
    toggleApiKeyBtn.addEventListener('click', () => {
      if (apiKeyInput.type === 'password') {
        apiKeyInput.type = 'text';
        toggleApiKeyBtn.textContent = 'Hide';
      } else {
        apiKeyInput.type = 'password';
        toggleApiKeyBtn.textContent = 'Show';
      }
    });
    
    // Test API button click handler
    testApiBtn.addEventListener('click', () => {
      const apiUrl = apiUrlInput.value.trim();
      const apiKey = apiKeyInput.value.trim();
      
      if (!apiUrl) {
        apiStatusDiv.textContent = 'Please enter an API URL';
        apiStatusDiv.style.color = '#ff9800';
        return;
      }
      
      if (!apiKey) {
        apiStatusDiv.textContent = 'Please enter an API key';
        apiStatusDiv.style.color = '#ff9800';
        return;
      }
      
      apiStatusDiv.textContent = 'Testing API connection...';
      apiStatusDiv.style.color = '#ff9800';
      
      chrome.runtime.sendMessage({
        action: 'testBotDetectionAPI',
        serverUrl: apiUrl,
        apiKey: apiKey
      }, (response) => {
        if (response.success) {
          apiStatusDiv.textContent = `API connection successful`;
          apiStatusDiv.style.color = '#4caf50';
        } else {
          apiStatusDiv.textContent = `API connection failed: ${response.error}`;
          apiStatusDiv.style.color = '#f44336';
        }
      });
    });
    
    // Save button click handler
    saveButton.addEventListener('click', () => {
      const enabled = enabledCheckbox.checked;
      const apiKey = apiKeyInput.value.trim();
      const apiUrl = apiUrlInput.value.trim();
      const markNames = markNamesCheckbox.checked;
      const enableReportMenu = enableReportMenuCheckbox.checked;
      
      // Validate inputs
      if (enabled) {
        if (!apiKey) {
          statusSpan.textContent = 'Error: API key is required';
          statusSpan.style.color = '#f44336';
          return;
        }
        
        if (!apiUrl) {
          statusSpan.textContent = 'Error: API URL is required';
          statusSpan.style.color = '#f44336';
          return;
        }
      }
      
      // Send to background script
      chrome.runtime.sendMessage({
        action: 'setResponseModification',
        enabled: enabled,
        apiKey: apiKey,
        serverUrl: apiUrl,
        markBotNames: markNames,
        enableReportMenu: enableReportMenu
      }, (response) => {
        if (response.success) {
          statusSpan.textContent = enabled ? 'Bot detection enabled!' : 'Bot detection disabled!';
          statusSpan.style.color = '#4caf50';
          
          // Save to storage
          chrome.storage.local.set({
            'botDetectionEnabled': enabled,
            'apiKey': apiKey,
            'apiServerUrl': apiUrl,
            'markBotNames': markNames,
            'enableReportMenu': enableReportMenu
          });
        } else {
          statusSpan.textContent = `Error: ${response.error}`;
          statusSpan.style.color = '#f44336';
        }
        
        // Clear status after 3 seconds
        setTimeout(() => {
          statusSpan.textContent = '';
        }, 3000);
      });
    });
  }, 500);
}

// Function to load response modification settings
function loadResponseModificationSettings() {
  chrome.runtime.sendMessage({
    action: 'getResponseModificationStatus'
  }, (response) => {
    if (!response) return;
    
    const enabledCheckbox = document.getElementById('yt-modify-enabled');
    const serverUrlInput = document.getElementById('yt-server-url');
    const apiKeyInput = document.getElementById('yt-modify-api-key');
    
    if (enabledCheckbox) {
      enabledCheckbox.checked = response.enabled;
    }
    
    if (serverUrlInput && response.serverUrl) {
      serverUrlInput.value = response.serverUrl;
    }

    // Get API key from storage
    chrome.storage.local.get('modifyApiKey', (data) => {
      if (apiKeyInput && data.modifyApiKey) {
        apiKeyInput.value = data.modifyApiKey;
      } else if (apiKeyInput && response.apiKey) {
        // Fall back to the general API key if no specific one is found
        apiKeyInput.value = response.apiKey;
      }
    });
    
    // Get function code from storage
    chrome.storage.local.get('responseModificationFunction', (data) => {
      const functionTextarea = document.getElementById('yt-modify-function');
      if (functionTextarea && data.responseModificationFunction) {
        functionTextarea.value = data.responseModificationFunction;
      }
    });
  });
}

// Function to load bot detection settings
function loadBotDetectionSettings() {
  chrome.runtime.sendMessage({
    action: 'getResponseModificationStatus'
  }, (response) => {
    if (!response) return;
    
    const enabledCheckbox = document.getElementById('yt-bot-detection-enabled');
    const apiKeyInput = document.getElementById('yt-api-key');
    const apiUrlInput = document.getElementById('yt-bot-api-url');
    const enableReportMenuCheckbox = document.getElementById('yt-enable-report-menu');
    
    if (enabledCheckbox) {
      chrome.storage.local.get('botDetectionEnabled', (data) => {
        enabledCheckbox.checked = data.botDetectionEnabled || false;
      });
    }
    
    if (apiKeyInput && response.apiKey) {
      apiKeyInput.value = response.apiKey;
    }
    
    if (apiUrlInput && response.serverUrl) {
      apiUrlInput.value = response.serverUrl;
    }
    
    // Get mark names setting
    chrome.storage.local.get('markBotNames', (data) => {
      const markNamesCheckbox = document.getElementById('yt-bot-mark-names');
      if (markNamesCheckbox) {
        markNamesCheckbox.checked = data.markBotNames !== false; // Default to true
      }
    });

    // Get enable report menu setting
    chrome.storage.local.get('enableReportMenu', (data) => {
      const enableReportMenuCheckbox = document.getElementById('yt-enable-report-menu');
      if (enableReportMenuCheckbox) {
        enableReportMenuCheckbox.checked = data.enableReportMenu !== false; // Default to true
      }
    });
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
    
    element.style.top = (element.offsetTop - pos2) + "px";
    element.style.left = (element.offsetLeft - pos1) + "px";
    element.style.bottom = "auto";
    element.style.right = "auto";
  }
  
  function closeDragElement() {
    document.onmouseup = null;
    document.onmousemove = null;
  }
}

// Function to update the requests list
function updateRequestsList() {
  const container = document.getElementById('yt-intercepted-requests');
  if (!container) return;
  
  // Get intercepted requests from background script
  chrome.runtime.sendMessage({ action: 'getInterceptedRequests' }, (response) => {
    if (!response || !response.requests) return;
    
    container.innerHTML = '';
    
    if (response.requests.length === 0) {
      container.innerHTML = '<p>No requests intercepted yet.</p>';
      return;
    }
    
    // Display the last 20 requests
    const recentRequests = response.requests.slice(-20).reverse();
    
    recentRequests.forEach((req, index) => {
      const reqElement = document.createElement('div');
      reqElement.style.cssText = `
        margin-bottom: 8px;
        padding-bottom: 8px;
        border-bottom: 1px solid #444;
      `;
      
      // Format URL to show only the path
      let displayUrl = req.url;
      try {
        const urlObj = new URL(req.url);
        displayUrl = urlObj.pathname + urlObj.search;
        if (displayUrl.length > 40) {
          displayUrl = displayUrl.substring(0, 37) + '...';
        }
      } catch (e) {}
      
      // Status code styling
      let statusStyle = '';
      let statusText = '';
      if (req.statusCode) {
        if (req.statusCode >= 200 && req.statusCode < 300) {
          statusStyle = 'color: #4caf50;'; // Green for success
        } else if (req.statusCode >= 400) {
          statusStyle = 'color: #f44336;'; // Red for error
        } else {
          statusStyle = 'color: #ff9800;'; // Orange for other
        }
        statusText = `<span style="${statusStyle}">${req.statusCode}</span>`;
      }
      
      // Modified indicator
      let modifiedBadge = '';
      if (req.modified) {
        let badgeColor = '#8e44ad'; // Purple for regular modifications
        
        // If it's from the server, use a different color
        if (req.modificationSource === 'server') {
          badgeColor = '#2980b9'; // Blue for server modifications
        }
        
        modifiedBadge = `<span style="background: ${badgeColor}; color: white; padding: 1px 4px; border-radius: 3px; font-size: 9px; margin-left: 5px;">MODIFIED</span>`;
        
        // If there was an error, show a warning
        if (req.modificationError) {
          modifiedBadge += `<span style="background: #e74c3c; color: white; padding: 1px 4px; border-radius: 3px; font-size: 9px; margin-left: 5px;">ERROR</span>`;
        }
        
        // If it's currently processing
        if (req.processing) {
          modifiedBadge += `<span style="background: #f39c12; color: white; padding: 1px 4px; border-radius: 3px; font-size: 9px; margin-left: 5px;">PROCESSING</span>`;
        }
      }
      
      // Bot detection badge
      if (req.botDetection) {
        modifiedBadge += `<span style="background: #27ae60; color: white; padding: 1px 4px; border-radius: 3px; font-size: 9px; margin-left: 5px;">BOT DETECTION</span>`;
      }
      
      reqElement.innerHTML = `
        <div><strong>${index + 1}. ${req.method}</strong> ${statusText} ${modifiedBadge} ${displayUrl}</div>
        <div style="font-size: 10px; color: #aaa;">${req.timestamp}</div>
      `;
      
      // Add click event to show full URL and details
      reqElement.addEventListener('click', () => {
        const details = [
          `URL: ${req.url}`,
          `Method: ${req.method}`,
          `Type: ${req.type}`,
          `Request Time: ${req.timestamp}`
        ];
        
        if (req.statusCode) {
          details.push(`Status Code: ${req.statusCode}`);
        }
        
        if (req.responseTime) {
          details.push(`Response Time: ${req.responseTime}`);
        }
        
        if (req.modified) {
          details.push(`Response Modified: Yes`);
          
          if (req.modificationSource) {
            details.push(`Modification Source: ${req.modificationSource}`);
          }
          
          if (req.modifiedTimestamp) {
            details.push(`Modified At: ${req.modifiedTimestamp}`);
          }
          
          if (req.modificationError) {
            details.push(`Modification Error: ${req.modificationError}`);
          }
        }
        
        if (req.botDetection) {
          details.push(`Bot Detection: Active`);
          if (req.botsDetected) {
            details.push(`Bots Detected: ${req.botsDetected}`);
          }
        }
        
        alert(details.join('\n'));
      });
      
      container.appendChild(reqElement);
    });
  });
}

// Wait for page to load before adding panel
window.addEventListener('load', () => {
  // Check if we're on YouTube
  if (window.location.hostname.includes('youtube.com')) {
    const { panel, requestsContainer } = addControlPanel();
    
    // Update requests list periodically
    setInterval(() => {
      if (requestsContainer.style.display !== 'none') {
        updateRequestsList();
      }
    }, 5000);
    
    // Set up comment report menu
    setupCommentReportMenu();
  }
});

// Function to set up the comment report menu
function setupCommentReportMenu() {
  // Check if report menu is enabled
  chrome.storage.local.get('enableReportMenu', (data) => {
    if (data.enableReportMenu === false) return;
    
    // Create custom context menu
    createCustomContextMenu();
    
    // Add event listener for right clicks on comments
    document.addEventListener('contextmenu', handleCommentRightClick);
  });
}

// Custom context menu for reporting comments
let contextMenu = null;
let currentCommentData = null;

// Function to create the custom context menu
function createCustomContextMenu() {
  // Create the context menu if it doesn't exist
  if (!contextMenu) {
    contextMenu = document.createElement('div');
    contextMenu.id = 'yt-report-context-menu';
    contextMenu.style.cssText = `
      position: absolute;
      background: rgba(33, 33, 33, 0.95);
      color: white;
      padding: 8px 0;
      border-radius: 4px;
      z-index: 10000;
      font-family: Arial, sans-serif;
      font-size: 14px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.5);
      display: none;
    `;
    
    // Add report option
    const reportOption = document.createElement('div');
    reportOption.textContent = 'Report as Bot';
    reportOption.style.cssText = `
      padding: 8px 16px;
      cursor: pointer;
    `;
    reportOption.addEventListener('mouseover', () => {
      reportOption.style.background = 'rgba(255, 255, 255, 0.1)';
    });
    reportOption.addEventListener('mouseout', () => {
      reportOption.style.background = 'transparent';
    });
    reportOption.addEventListener('click', () => {
      if (currentCommentData) {
        reportComment(currentCommentData);
      }
      hideContextMenu();
    });
    
    contextMenu.appendChild(reportOption);
    document.body.appendChild(contextMenu);
    
    // Hide context menu when clicking elsewhere
    document.addEventListener('click', hideContextMenu);
  }
}

// Function to handle right clicks on comments
function handleCommentRightClick(event) {
  // Check if we're on a comment
  const commentElement = findCommentElement(event.target);
  if (!commentElement) return;
  
  // Extract comment data
  const commentData = extractCommentData(commentElement);
  if (!commentData) return;
  
  // Store the current comment data
  currentCommentData = commentData;
  
  // Show the context menu
  showContextMenu(event.clientX, event.clientY);
  
  // Prevent the default context menu
  event.preventDefault();
}

// Function to find the comment element from a clicked element
function findCommentElement(element) {
  // YouTube comment structure changes frequently, so we need to be flexible
  // Look for common comment container classes or attributes
  
  // Try to find the comment container by traversing up the DOM
  let current = element;
  const maxDepth = 10; // Prevent infinite loops
  let depth = 0;
  
  while (current && depth < maxDepth) {
    // Check if this is a comment element
    // YouTube comments are typically in ytd-comment-renderer or similar elements
    if (
      current.tagName && (
        current.tagName.toLowerCase() === 'ytd-comment-renderer' ||
        current.tagName.toLowerCase() === 'ytd-comment-thread-renderer' ||
        (current.id && current.id.includes('comment')) ||
        (current.className && typeof current.className === 'string' && 
         (current.className.includes('comment') || current.className.includes('Comment')))
      )
    ) {
      return current;
    }
    
    current = current.parentElement;
    depth++;
  }
  
  return null;
}

// Function to extract comment data from a comment element
function extractCommentData(commentElement) {
  try {
    // Try to find the author name
    let authorElement = commentElement.querySelector('#author-text') || 
                       commentElement.querySelector('.ytd-comment-renderer #author-text') ||
                       commentElement.querySelector('[id*="author"]');
    
    // Try to find the comment content
    let contentElement = commentElement.querySelector('#content-text') || 
                        commentElement.querySelector('.ytd-comment-renderer #content-text') ||
                        commentElement.querySelector('[id*="content"]');
    
    if (!authorElement || !contentElement) return null;
    
    const authorName = authorElement.textContent.trim();
    const content = contentElement.textContent.trim();
    
    if (!authorName || !content) return null;
    
    return {
      author_name: authorName,
      content: content,
      element: commentElement
    };
  } catch (error) {
    console.error('Error extracting comment data:', error);
    return null;
  }
}

// Function to show the context menu
function showContextMenu(x, y) {
  if (!contextMenu) return;
  
  // Position the menu
  contextMenu.style.left = `${x}px`;
  contextMenu.style.top = `${y}px`;
  contextMenu.style.display = 'block';
}

// Function to hide the context menu
function hideContextMenu() {
  if (!contextMenu) return;
  contextMenu.style.display = 'none';
  currentCommentData = null;
}

// Function to report a comment to the API
function reportComment(commentData) {
  // Get API settings
  chrome.storage.local.get(['apiKey', 'apiServerUrl'], (data) => {
    if (!data.apiKey || !data.apiServerUrl) {
      alert('API key or server URL not configured. Please configure them in the Bot Detection settings.');
      return;
    }
    
    // Show reporting status
    const statusElement = document.createElement('div');
    statusElement.style.cssText = `
      position: fixed;
      bottom: 20px;
      left: 20px;
      background: rgba(33, 33, 33, 0.9);
      color: white;
      padding: 10px;
      border-radius: 5px;
      z-index: 10000;
      font-family: Arial, sans-serif;
      box-shadow: 0 0 10px rgba(0,0,0,0.5);
    `;
    statusElement.textContent = 'Reporting comment...';
    document.body.appendChild(statusElement);
    
    // Send the report via background script
    chrome.runtime.sendMessage({
      action: 'reportComment',
      serverUrl: data.apiServerUrl,
      apiKey: data.apiKey,
      comment: {
        author_name: commentData.author_name,
        content: commentData.content,
        is_bot: true
      }
    }, (response) => {
      if (response && response.success) {
        // Success - update status
        statusElement.textContent = 'Comment reported successfully!';
        statusElement.style.background = 'rgba(46, 125, 50, 0.9)'; // Green
        
        // Add visual indicator to the reported comment
        if (commentData.element) {
          const reportBadge = document.createElement('span');
          reportBadge.textContent = 'REPORTED';
          reportBadge.style.cssText = `
            background: #e74c3c;
            color: white;
            padding: 2px 5px;
            border-radius: 3px;
            font-size: 10px;
            margin-left: 5px;
            vertical-align: middle;
          `;
          
          // Try to add the badge to the author element
          try {
            const authorElement = commentData.element.querySelector('#author-text') || 
                                 commentData.element.querySelector('.ytd-comment-renderer #author-text') ||
                                 commentData.element.querySelector('[id*="author"]');
            if (authorElement) {
              authorElement.appendChild(reportBadge);
            }
          } catch (e) {
            console.error('Error adding report badge:', e);
          }
        }
      } else {
        // Error - update status
        statusElement.textContent = `Error: ${response && response.error ? response.error : 'Failed to report comment'}`;
        statusElement.style.background = 'rgba(198, 40, 40, 0.9)'; // Red
      }
      
      // Remove status after a delay
      setTimeout(() => {
        document.body.removeChild(statusElement);
      }, 3000);
    });
  });
}

console.log('YouTube Request Interceptor content script loaded'); 