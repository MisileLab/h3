document.addEventListener('DOMContentLoaded', async () => {
  // DOM Elements
  const apiKeyInput = document.getElementById('apiKeyInput');
  const toggleVisibilityBtn = document.getElementById('toggleVisibility');
  const saveBtn = document.getElementById('saveBtn');
  const deleteBtn = document.getElementById('deleteBtn');
  const statusMessage = document.getElementById('statusMessage');

  // Load saved API key
  const apiKey = await Storage.getApiKey();
  if (apiKey) {
    apiKeyInput.value = apiKey;
  }

  // Toggle password visibility
  toggleVisibilityBtn.addEventListener('click', () => {
    if (apiKeyInput.type === 'password') {
      apiKeyInput.type = 'text';
      toggleVisibilityBtn.textContent = 'Hide';
    } else {
      apiKeyInput.type = 'password';
      toggleVisibilityBtn.textContent = 'Show';
    }
  });

  // Save API key
  saveBtn.addEventListener('click', async () => {
    const apiKey = apiKeyInput.value.trim();
    
    if (!apiKey) {
      showStatus('Please enter an API key', 'error');
      return;
    }

    try {
      await Storage.setApiKey(apiKey);
      showStatus('API key saved successfully', 'success');
    } catch (error) {
      showStatus('Failed to save API key', 'error');
    }
  });

  // Delete API key
  deleteBtn.addEventListener('click', async () => {
    try {
      await Storage.removeApiKey();
      apiKeyInput.value = '';
      showStatus('API key deleted successfully', 'success');
    } catch (error) {
      showStatus('Failed to delete API key', 'error');
    }
  });

  // Show status message
  function showStatus(message, type) {
    statusMessage.textContent = message;
    statusMessage.className = 'status-message';
    statusMessage.classList.add(type);

    setTimeout(() => {
      statusMessage.className = 'status-message';
    }, 3000);
  }
});