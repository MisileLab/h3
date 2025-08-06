document.addEventListener('DOMContentLoaded', async () => {
  // DOM Elements
  const videoIdInput = document.getElementById('videoIdInput');
  const uploadBtn = document.getElementById('uploadBtn');
  const uploadStatus = document.getElementById('uploadStatus');
  const checkStatusBtn = document.getElementById('checkStatusBtn');
  const embeddingStatus = document.getElementById('embeddingStatus');
  const searchInput = document.getElementById('searchInput');
  const searchBtn = document.getElementById('searchBtn');
  const searchResults = document.getElementById('searchResults');
  const loadingIndicator = document.getElementById('loadingIndicator');
  const errorToast = document.getElementById('errorToast');
  const apiKeyWarning = document.getElementById('apiKeyWarning');
  const openOptionsBtn = document.getElementById('openOptions');

  // Uploaded video IDs
  let uploadedVideoIds = [];

  // Check if API key exists
  const apiKey = await Storage.getApiKey();
  if (!apiKey) {
    apiKeyWarning.classList.remove('hidden');
  }

  // Open options page
  openOptionsBtn.addEventListener('click', () => {
    chrome.runtime.openOptionsPage();
  });

  // Upload video
  uploadBtn.addEventListener('click', async () => {
    const videoId = videoIdInput.value.trim();
    if (!videoId) {
      showError('Please enter a video ID');
      return;
    }

    const apiKey = await Storage.getApiKey();
    if (!apiKey) {
      showError('API key not set. Please set it in options.');
      return;
    }

    showLoading(true);
    try {
      await Api.uploadVideo(videoId, apiKey);
      uploadStatus.textContent = `Video ${videoId} uploaded successfully`;
      uploadStatus.style.color = 'var(--success-color)';
      videoIdInput.value = '';
      
      // Add to uploaded videos list
      uploadedVideoIds.push(videoId);
    } catch (error) {
      showError(error.message || 'Failed to upload video');
      uploadStatus.textContent = 'Upload failed';
      uploadStatus.style.color = 'var(--error-color)';
    } finally {
      showLoading(false);
    }
  });

  // Check embedding status
  checkStatusBtn.addEventListener('click', async () => {
    if (uploadedVideoIds.length === 0) {
      showError('No videos uploaded yet');
      return;
    }

    const apiKey = await Storage.getApiKey();
    if (!apiKey) {
      showError('API key not set. Please set it in options.');
      return;
    }

    showLoading(true);
    try {
      const status = await Api.checkEmbeddingStatus(uploadedVideoIds, apiKey);
      embeddingStatus.textContent = `Ready videos: ${status.readyCount}/${uploadedVideoIds.length}`;
      embeddingStatus.style.color = status.allReady ? 'var(--success-color)' : 'var(--warning-color)';
    } catch (error) {
      showError(error.message || 'Failed to check embedding status');
      embeddingStatus.textContent = 'Status check failed';
      embeddingStatus.style.color = 'var(--error-color)';
    } finally {
      showLoading(false);
    }
  });

  // Search videos
  searchBtn.addEventListener('click', async () => {
    const query = searchInput.value.trim();
    if (!query) {
      showError('Please enter a search query');
      return;
    }

    const apiKey = await Storage.getApiKey();
    if (!apiKey) {
      showError('API key not set. Please set it in options.');
      return;
    }

    showLoading(true);
    try {
      const results = await Api.searchVideos(query, apiKey);
      displaySearchResults(results);
    } catch (error) {
      showError(error.message || 'Search failed');
      searchResults.innerHTML = '<p>No results found</p>';
    } finally {
      showLoading(false);
    }
  });

  // Display search results
  function displaySearchResults(results) {
    searchResults.innerHTML = '';
    
    if (!results || results.length === 0) {
      searchResults.innerHTML = '<p>No results found</p>';
      return;
    }

    results.forEach(result => {
      const resultItem = document.createElement('div');
      resultItem.className = 'result-item';
      
      resultItem.innerHTML = `
        <div class="result-thumbnail">
          <img src="https://i.ytimg.com/vi/${result.videoId}/mqdefault.jpg" alt="${result.title}" width="120">
        </div>
        <div class="result-info">
          <div class="result-title">${result.title}</div>
          <div class="result-id">${result.videoId}</div>
        </div>
      `;
      
      searchResults.appendChild(resultItem);
    });
  }

  // Show loading indicator
  function showLoading(isLoading) {
    if (isLoading) {
      loadingIndicator.classList.remove('hidden');
    } else {
      loadingIndicator.classList.add('hidden');
    }
  }

  // Show error toast
  function showError(message) {
    errorToast.textContent = message;
    errorToast.classList.remove('hidden');
    
    setTimeout(() => {
      errorToast.classList.add('hidden');
    }, 3000);
  }
});