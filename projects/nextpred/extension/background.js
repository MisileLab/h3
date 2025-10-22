/**
 * Background Service Worker for Next Action Predictor
 * Handles event tracking, data collection, and model inference
 */

import { StorageManager } from './utils/storage.js';
import { APIManager } from './utils/api.js';
import { InferenceEngine } from './utils/inference.js';

class BackgroundService {
  constructor() {
    this.storage = new StorageManager();
    this.api = new APIManager();
    this.inference = new InferenceEngine();
    this.uploadInterval = 5 * 60 * 1000; // 5 minutes
    this.modelUpdateInterval = 24 * 60 * 60 * 1000; // 24 hours
    
    this.init();
  }

  async init() {
    console.log('Next Action Predictor: Background service initialized');
    
    // Set up event listeners
    this.setupEventListeners();
    
    // Start periodic tasks
    this.startPeriodicTasks();
    
    // Load or download model
    await this.initializeModel();
  }

  setupEventListeners() {
    // Tab activation events
    chrome.tabs.onActivated.addListener(async (activeInfo) => {
      await this.handleTabSwitch(activeInfo.tabId, activeInfo.windowId);
    });

    // Tab update events (URL changes, loading complete)
    chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
      if (changeInfo.status === 'complete' && tab.url) {
        await this.handleNavigation(tabId, tab.url, tab.title);
      }
    });

    // Web navigation events
    chrome.webNavigation.onCompleted.addListener(async (details) => {
      if (details.frameId === 0) { // Main frame only
        await this.handlePageLoad(details.tabId, details.url, details.timeStamp);
      }
    });

    // Command listener for Ctrl+Space
    chrome.commands.onCommand.addListener(async (command) => {
      if (command === 'show-predictions') {
        await this.showPredictions();
      }
    });

    // Extension install/update
    chrome.runtime.onInstalled.addListener(async (details) => {
      if (details.reason === 'install') {
        await this.storage.initializeStorage();
        console.log('Next Action Predictor: Extension installed');
      }
    });

    // Message handlers for popup and content scripts
    chrome.runtime.onMessage.addListener(async (message, sender, sendResponse) => {
      try {
        switch (message.action) {
          case 'getPredictions':
            await this.handleGetPredictions(message, sender, sendResponse);
            break;
          case 'recordFeedback':
            await this.handleRecordFeedback(message, sender, sendResponse);
            break;
          case 'getModelInfo':
            await this.handleGetModelInfo(message, sender, sendResponse);
            break;
          case 'userFeedback':
            await this.handleUserFeedback(message, sender, sendResponse);
            break;
          default:
            console.warn('Unknown message action:', message.action);
            sendResponse({ error: 'Unknown action' });
        }
      } catch (error) {
        console.error('Error handling message:', error);
        sendResponse({ error: error.message });
      }
      return true; // Keep message channel open for async response
    });
  }

  async handleTabSwitch(tabId, windowId) {
    try {
      const tab = await chrome.tabs.get(tabId);
      if (!tab.url || this.isSensitiveUrl(tab.url)) return;

      const event = {
        type: 'tab_switch',
        timestamp: Date.now(),
        data: {
          tabId,
          windowId,
          url: tab.url,
          title: tab.title,
          index: tab.index,
          active: tab.active
        }
      };

      await this.storage.addEvent(event);
    } catch (error) {
      console.error('Error handling tab switch:', error);
    }
  }

  async handleNavigation(tabId, url, title) {
    try {
      if (this.isSensitiveUrl(url)) return;

      const event = {
        type: 'navigation',
        timestamp: Date.now(),
        data: {
          tabId,
          url,
          title,
          urlDomain: this.extractDomain(url)
        }
      };

      await this.storage.addEvent(event);
    } catch (error) {
      console.error('Error handling navigation:', error);
    }
  }

  async handlePageLoad(tabId, url, timestamp) {
    try {
      if (this.isSensitiveUrl(url)) return;

      // Get scroll position from content script
      const scrollData = await this.getScrollPosition(tabId);
      
      const event = {
        type: 'page_load',
        timestamp,
        data: {
          tabId,
          url,
          urlDomain: this.extractDomain(url),
          scrollPosition: scrollData?.position || 0,
          pageHeight: scrollData?.height || 0
        }
      };

      await this.storage.addEvent(event);
    } catch (error) {
      console.error('Error handling page load:', error);
    }
  }

  async getScrollPosition(tabId) {
    try {
      const response = await chrome.tabs.sendMessage(tabId, { 
        action: 'getScrollPosition' 
      });
      return response;
    } catch (error) {
      // Content script might not be loaded yet
      return null;
    }
  }

  async showPredictions() {
    try {
      // Get current context
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (!tab) return;

      // Get recent events for context
      const context = await this.storage.getRecentContext(10 * 60 * 1000); // 10 minutes
      
      // Run inference
      const predictions = await this.inference.predict(context, tab);
      
      // Show popup with predictions
      await this.openPredictionPopup(predictions, tab);
      
    } catch (error) {
      console.error('Error showing predictions:', error);
    }
  }

  async openPredictionPopup(predictions, currentTab) {
    try {
      // Store predictions temporarily for popup
      await chrome.storage.local.set({
        currentPredictions: predictions,
        currentTab: currentTab,
        predictionTimestamp: Date.now()
      });

      // Open popup
      chrome.action.openPopup();
    } catch (error) {
      console.error('Error opening prediction popup:', error);
    }
  }

  async initializeModel() {
    try {
      const modelVersion = await this.api.getModelVersion();
      const localVersion = await this.storage.getModelVersion();

      if (!localVersion || localVersion !== modelVersion.version) {
        console.log('Downloading new model version:', modelVersion.version);
        await this.inference.downloadModel(modelVersion.version);
        await this.storage.setModelVersion(modelVersion.version);
      } else {
        console.log('Using cached model version:', localVersion);
        await this.inference.loadModel();
      }
    } catch (error) {
      console.error('Error initializing model:', error);
      // Continue without model - will use rule-based predictions
    }
  }

  startPeriodicTasks() {
    // Upload events every 5 minutes
    setInterval(async () => {
      await this.uploadEvents();
    }, this.uploadInterval);

    // Check for model updates every 24 hours
    setInterval(async () => {
      await this.initializeModel();
    }, this.modelUpdateInterval);
  }

  async uploadEvents() {
    try {
      const events = await this.storage.getEventsForUpload();
      if (events.length === 0) return;

      const response = await this.api.uploadEvents(events);
      if (response.success) {
        await this.storage.markEventsUploaded(events.map(e => e.id));
        console.log(`Uploaded ${events.length} events`);
      }
    } catch (error) {
      console.error('Error uploading events:', error);
    }
  }

  isSensitiveUrl(url) {
    const sensitivePatterns = [
      /accounts\./,
      /auth\./,
      /login/,
      /token=/,
      /session=/,
      /key=/,
      /password/,
      /bank/,
      /payment/
    ];

    return sensitivePatterns.some(pattern => pattern.test(url));
  }

  extractDomain(url) {
    try {
      return new URL(url).hostname;
    } catch {
      return '';
    }
  }

  // Message Handlers
  async handleGetPredictions(message, sender, sendResponse) {
    try {
      // Get current tab
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (!tab) {
        sendResponse({ predictions: [] });
        return;
      }

      // Get recent events for context
      const context = await this.storage.getRecentContext(10 * 60 * 1000); // 10 minutes
      
      // Run inference
      const predictions = await this.inference.predict(context, tab);
      
      // Store predictions for popup
      await chrome.storage.local.set({
        currentPredictions: predictions,
        currentTab: tab,
        predictionTimestamp: Date.now()
      });

      sendResponse({ predictions });
    } catch (error) {
      console.error('Error getting predictions:', error);
      sendResponse({ predictions: [] });
    }
  }

  async handleRecordFeedback(message, sender, sendResponse) {
    try {
      const feedback = {
        ...message.feedback,
        timestamp: Date.now(),
        userAgent: navigator.userAgent
      };

      // Store feedback locally
      await this.storage.addFeedback(feedback);

      // Try to upload to server immediately
      try {
        const response = await this.api.recordFeedback(feedback);
        if (response.success) {
          await this.storage.markFeedbackUploaded(feedback.id);
        }
      } catch (uploadError) {
        console.warn('Failed to upload feedback immediately, will retry later:', uploadError);
      }

      sendResponse({ success: true, feedbackId: feedback.id });
    } catch (error) {
      console.error('Error recording feedback:', error);
      sendResponse({ success: false, error: error.message });
    }
  }

  async handleGetModelInfo(message, sender, sendResponse) {
    try {
      const modelVersion = await this.storage.getModelVersion();
      const modelLoaded = await this.inference.isModelLoaded();
      
      sendResponse({
        loaded: modelLoaded,
        version: modelVersion || 'unknown',
        lastUpdated: await this.storage.getModelLastUpdated()
      });
    } catch (error) {
      console.error('Error getting model info:', error);
      sendResponse({
        loaded: false,
        version: 'unknown',
        error: error.message
      });
    }
  }

  async handleUserFeedback(message, sender, sendResponse) {
    try {
      const feedback = {
        type: 'user_feedback',
        subtype: message.feedback.type || 'general',
        content: message.feedback.content,
        timestamp: message.feedback.timestamp || Date.now(),
        source: 'popup'
      };

      // Store feedback
      await this.storage.addFeedback(feedback);

      // Try to upload immediately
      try {
        await this.api.recordFeedback(feedback);
      } catch (uploadError) {
        console.warn('Failed to upload user feedback immediately:', uploadError);
      }

      sendResponse({ success: true });
    } catch (error) {
      console.error('Error handling user feedback:', error);
      sendResponse({ success: false, error: error.message });
    }
  }
}

// Initialize background service
const backgroundService = new BackgroundService();