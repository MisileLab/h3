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
  
  // Toggle display of requests
  toggleBtn.addEventListener('click', () => {
    if (requestsContainer.style.display === 'none') {
      requestsContainer.style.display = 'block';
      modifyContainer.style.display = 'none';
      toggleBtn.textContent = 'Hide Requests';
      modifyBtn.style.background = '#555';
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
      modifyBtn.style.background = '#065fd4';
      toggleBtn.style.background = '#555';
      toggleBtn.textContent = 'Show Requests';
      loadResponseModificationSettings();
    } else {
      modifyContainer.style.display = 'none';
      modifyBtn.style.background = '#555';
    }
  });
  
  header.appendChild(toggleBtn);
  header.appendChild(modifyBtn);
  panel.appendChild(requestsContainer);
  panel.appendChild(modifyContainer);
  
  // Add panel to page
  document.body.appendChild(panel);
  
  // Make panel draggable
  makeDraggable(panel, header);
  
  // Set up response modification events
  setupResponseModification();
  
  return { panel, requestsContainer, modifyContainer };
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
    const saveButton = document.getElementById('yt-modify-save');
    const statusSpan = document.getElementById('yt-modify-status');
    
    if (!enabledCheckbox || !functionTextarea || !saveButton || !statusSpan || !serverUrlInput || !testServerBtn) return;
    
    // Test server button click handler
    testServerBtn.addEventListener('click', () => {
      const serverUrl = serverUrlInput.value.trim();
      
      if (!serverUrl) {
        serverStatusDiv.textContent = 'Please enter a server URL';
        serverStatusDiv.style.color = '#ff9800';
        return;
      }
      
      serverStatusDiv.textContent = 'Testing connection...';
      serverStatusDiv.style.color = '#ff9800';
      
      chrome.runtime.sendMessage({
        action: 'testServerConnection',
        serverUrl: serverUrl
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
        serverUrl: serverUrl
      }, (response) => {
        if (response.success) {
          statusSpan.textContent = enabled ? 'Enabled!' : 'Disabled!';
          statusSpan.style.color = '#4caf50';
          
          // Save to storage
          chrome.storage.local.set({
            'responseModificationEnabled': enabled,
            'responseModificationFunction': functionCode,
            'exampleServerUrl': serverUrl
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
    
    if (enabledCheckbox) {
      enabledCheckbox.checked = response.enabled;
    }
    
    if (serverUrlInput && response.serverUrl) {
      serverUrlInput.value = response.serverUrl;
    }
    
    // Get function code from storage
    chrome.storage.local.get('responseModificationFunction', (data) => {
      const functionTextarea = document.getElementById('yt-modify-function');
      if (functionTextarea && data.responseModificationFunction) {
        functionTextarea.value = data.responseModificationFunction;
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
  }
});

console.log('YouTube Request Interceptor content script loaded'); 