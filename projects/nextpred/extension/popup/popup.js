/**
 * Popup Script for Next Action Predictor
 * Displays predictions and handles user interactions
 */

class PopupController {
  constructor() {
    this.predictions = [];
    this.selectedIndex = 0;
    this.currentTab = null;
    
    this.init();
  }

  async init() {
    // Load predictions and current tab from storage
    const data = await chrome.storage.local.get([
      'currentPredictions', 
      'currentTab', 
      'predictionTimestamp'
    ]);

    this.predictions = data.currentPredictions || [];
    this.currentTab = data.currentTab;
    this.predictionTimestamp = data.predictionTimestamp;

    // Check if predictions are stale (older than 30 seconds)
    if (this.isPredictionsStale()) {
      this.showLoading();
      await this.requestNewPredictions();
    } else {
      this.renderPredictions();
    }

    this.setupEventListeners();
    this.updateModelStatus();
  }

  isPredictionsStale() {
    if (!this.predictionTimestamp) return true;
    const age = Date.now() - this.predictionTimestamp;
    return age > 30000; // 30 seconds
  }

  async requestNewPredictions() {
    try {
      // Request new predictions from background script
      const response = await chrome.runtime.sendMessage({
        action: 'getPredictions'
      });

      if (response && response.predictions) {
        this.predictions = response.predictions;
        this.predictionTimestamp = Date.now();
        await chrome.storage.local.set({
          currentPredictions: this.predictions,
          predictionTimestamp: this.predictionTimestamp
        });
        this.renderPredictions();
      } else {
        this.showNoPredictions();
      }
    } catch (error) {
      console.error('Error requesting predictions:', error);
      this.showNoPredictions();
    }
  }

  renderPredictions() {
    const container = document.getElementById('predictionsContainer');
    const noPredictions = document.getElementById('noPredictions');
    const loading = document.getElementById('loading');

    loading.style.display = 'none';

    if (!this.predictions || this.predictions.length === 0) {
      container.style.display = 'none';
      noPredictions.style.display = 'block';
      return;
    }

    container.style.display = 'flex';
    noPredictions.style.display = 'none';

    container.innerHTML = '';

    this.predictions.forEach((prediction, index) => {
      const item = this.createPredictionItem(prediction, index);
      container.appendChild(item);
    });

    // Select first item by default
    this.selectPrediction(0);
  }

  createPredictionItem(prediction, index) {
    const item = document.createElement('div');
    item.className = 'prediction-item';
    item.dataset.index = index;

    // Create favicon
    const favicon = document.createElement('img');
    favicon.className = 'prediction-favicon';
    favicon.src = `https://www.google.com/s2/favicons?domain=${new URL(prediction.url).hostname}&sz=16`;
    favicon.onerror = () => {
      favicon.src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><rect width="16" height="16" fill="%23ddd"/><text x="8" y="12" text-anchor="middle" font-size="8" fill="%23999">🌐</text></svg>';
    };

    // Create content
    const content = document.createElement('div');
    content.className = 'prediction-content';

    const title = document.createElement('div');
    title.className = 'prediction-title';
    title.textContent = prediction.title || 'Untitled';

    const url = document.createElement('div');
    url.className = 'prediction-url';
    url.textContent = new URL(prediction.url).hostname;

    content.appendChild(title);
    content.appendChild(url);

    // Create confidence
    const confidence = document.createElement('div');
    confidence.className = 'prediction-confidence';
    confidence.textContent = `${Math.round(prediction.confidence * 100)}%`;

    // Create number
    const number = document.createElement('div');
    number.className = 'prediction-number';
    number.textContent = index + 1;

    item.appendChild(number);
    item.appendChild(favicon);
    item.appendChild(content);
    item.appendChild(confidence);

    // Add click handler
    item.addEventListener('click', () => {
      this.selectPrediction(index);
      this.activatePrediction(prediction);
    });

    return item;
  }

  selectPrediction(index) {
    if (index < 0 || index >= this.predictions.length) return;

    // Remove previous selection
    document.querySelectorAll('.prediction-item').forEach(item => {
      item.classList.remove('selected', 'keyboard-selected');
    });

    // Add selection to current item
    const currentItem = document.querySelector(`[data-index="${index}"]`);
    if (currentItem) {
      currentItem.classList.add('selected');
    }

    this.selectedIndex = index;
  }

  async activatePrediction(prediction) {
    try {
      // Record feedback
      await this.recordFeedback(prediction, this.selectedIndex);

      // Switch to the predicted tab or navigate
      if (prediction.tabId) {
        // Switch to existing tab
        await chrome.tabs.update(prediction.tabId, { active: true });
      } else {
        // Create new tab or navigate to URL
        await chrome.tabs.create({ url: prediction.url });
      }

      // Close popup
      window.close();
    } catch (error) {
      console.error('Error activating prediction:', error);
    }
  }

  async recordFeedback(prediction, selectedIndex) {
    try {
      await chrome.runtime.sendMessage({
        action: 'recordFeedback',
        feedback: {
          predictionId: prediction.id,
          selectedIndex,
          timestamp: Date.now(),
          context: {
            currentTab: this.currentTab,
            allPredictions: this.predictions
          }
        }
      });
    } catch (error) {
      console.error('Error recording feedback:', error);
    }
  }

  setupEventListeners() {
    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
      switch (e.key) {
        case 'ArrowUp':
          e.preventDefault();
          this.selectPrediction(this.selectedIndex - 1);
          this.addKeyboardSelection();
          break;
        case 'ArrowDown':
          e.preventDefault();
          this.selectPrediction(this.selectedIndex + 1);
          this.addKeyboardSelection();
          break;
        case 'Enter':
          e.preventDefault();
          if (this.predictions[this.selectedIndex]) {
            this.activatePrediction(this.predictions[this.selectedIndex]);
          }
          break;
        case '1':
        case '2':
        case '3':
          e.preventDefault();
          const index = parseInt(e.key) - 1;
          if (this.predictions[index]) {
            this.selectPrediction(index);
            this.activatePrediction(this.predictions[index]);
          }
          break;
        case 'Escape':
          e.preventDefault();
          window.close();
          break;
      }
    });

    // Feedback button
    document.getElementById('feedbackBtn').addEventListener('click', () => {
      this.openFeedbackForm();
    });

    // Settings button
    document.getElementById('settingsBtn').addEventListener('click', () => {
      chrome.runtime.openOptionsPage();
    });
  }

  addKeyboardSelection() {
    const currentItem = document.querySelector(`[data-index="${this.selectedIndex}"]`);
    if (currentItem) {
      currentItem.classList.add('keyboard-selected');
    }
  }

  showLoading() {
    document.getElementById('loading').style.display = 'block';
    document.getElementById('predictionsContainer').style.display = 'none';
    document.getElementById('noPredictions').style.display = 'none';
  }

  showNoPredictions() {
    document.getElementById('loading').style.display = 'none';
    document.getElementById('predictionsContainer').style.display = 'none';
    document.getElementById('noPredictions').style.display = 'block';
  }

  async updateModelStatus() {
    const statusElement = document.getElementById('modelStatus');
    const indicator = statusElement.querySelector('.status-indicator');
    const text = statusElement.querySelector('.status-text');

    try {
      const modelInfo = await chrome.runtime.sendMessage({
        action: 'getModelInfo'
      });

      if (modelInfo && modelInfo.loaded) {
        indicator.className = 'status-indicator';
        text.textContent = `Model v${modelInfo.version}`;
      } else {
        indicator.className = 'status-indicator error';
        text.textContent = 'Model not loaded';
      }
    } catch (error) {
      indicator.className = 'status-indicator error';
      text.textContent = 'Status unknown';
    }
  }

  openFeedbackForm() {
    // Simple feedback form
    const feedback = prompt('How can we improve the predictions?');
    if (feedback) {
      chrome.runtime.sendMessage({
        action: 'userFeedback',
        feedback: {
          type: 'general',
          content: feedback,
          timestamp: Date.now()
        }
      });
    }
  }
}

// Initialize popup when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  new PopupController();
});