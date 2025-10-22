/**
 * Storage Manager for Next Action Predictor
 * Handles local storage of events and context data
 */

export class StorageManager {
  constructor() {
    this.storageKey = 'nextpred_data';
    this.maxEvents = 1000; // Maximum events to keep in memory
    this.contextWindowMs = 10 * 60 * 1000; // 10 minutes
  }

  async initializeStorage() {
    const data = await this.getStorageData();
    if (!data.events) {
      data.events = [];
      data.uploadedEvents = new Set();
      data.modelVersion = null;
      data.settings = {
        trackingEnabled: true,
        uploadInterval: 5 * 60 * 1000, // 5 minutes
        maxEvents: 1000
      };
      await this.setStorageData(data);
    }
  }

  async getStorageData() {
    const result = await chrome.storage.local.get(this.storageKey);
    return result[this.storageKey] || {
      events: [],
      uploadedEvents: [],
      modelVersion: null,
      settings: {}
    };
  }

  async setStorageData(data) {
    await chrome.storage.local.set({ [this.storageKey]: data });
  }

  async addEvent(event) {
    const data = await this.getStorageData();
    
    // Add unique ID and timestamp
    event.id = this.generateId();
    event.timestamp = event.timestamp || Date.now();
    
    // Add to events array
    data.events.unshift(event);
    
    // Limit array size
    if (data.events.length > data.settings.maxEvents) {
      data.events = data.events.slice(0, data.settings.maxEvents);
    }
    
    await this.setStorageData(data);
  }

  async getRecentContext(windowMs = this.contextWindowMs) {
    const data = await this.getStorageData();
    const cutoffTime = Date.now() - windowMs;
    
    // Filter events within the time window
    const recentEvents = data.events.filter(event => 
      event.timestamp > cutoffTime
    );

    // Build context object
    const context = {
      events: recentEvents,
      summary: this.summarizeEvents(recentEvents),
      timestamp: Date.now(),
      windowSize: windowMs
    };

    return context;
  }

  summarizeEvents(events) {
    const summary = {
      totalEvents: events.length,
      tabSwitches: 0,
      navigations: 0,
      searches: 0,
      scrolls: 0,
      clicks: 0,
      uniqueDomains: new Set(),
      recentUrls: [],
      recentSearches: [],
      timeDistribution: {}
    };

    events.forEach(event => {
      switch (event.type) {
        case 'tab_switch':
          summary.tabSwitches++;
          break;
        case 'navigation':
          summary.navigations++;
          if (event.data.url) {
            summary.uniqueDomains.add(this.extractDomain(event.data.url));
            if (summary.recentUrls.length < 5) {
              summary.recentUrls.push(event.data.url);
            }
          }
          break;
        case 'search':
          summary.searches++;
          if (event.data.query && summary.recentSearches.length < 3) {
            summary.recentSearches.push(event.data.query);
          }
          break;
        case 'scroll':
          summary.scrolls++;
          break;
        case 'click':
          summary.clicks++;
          break;
      }

      // Time distribution
      const hour = new Date(event.timestamp).getHours();
      summary.timeDistribution[hour] = (summary.timeDistribution[hour] || 0) + 1;
    });

    summary.uniqueDomains = Array.from(summary.uniqueDomains);
    return summary;
  }

  async getEventsForUpload() {
    const data = await this.getStorageData();
    
    // Get events that haven't been uploaded
    const eventsToUpload = data.events.filter(event => 
      !data.uploadedEvents.includes(event.id)
    );

    return eventsToUpload;
  }

  async markEventsUploaded(eventIds) {
    const data = await this.getStorageData();
    
    // Add to uploaded set
    eventIds.forEach(id => {
      data.uploadedEvents.push(id);
    });

    // Clean up old uploaded events (keep last 500)
    if (data.uploadedEvents.length > 500) {
      data.uploadedEvents = data.uploadedEvents.slice(-500);
    }

    await this.setStorageData(data);
  }

  async getModelVersion() {
    const data = await this.getStorageData();
    return data.modelVersion;
  }

  async setModelVersion(version) {
    const data = await this.getStorageData();
    data.modelVersion = version;
    await this.setStorageData(data);
  }

  async getSettings() {
    const data = await this.getStorageData();
    return data.settings;
  }

  async updateSettings(newSettings) {
    const data = await this.getStorageData();
    data.settings = { ...data.settings, ...newSettings };
    await this.setStorageData(data);
  }

  async clearOldData() {
    const data = await this.getStorageData();
    const cutoffTime = Date.now() - (24 * 60 * 60 * 1000); // 24 hours
    
    // Remove old events
    data.events = data.events.filter(event => event.timestamp > cutoffTime);
    
    // Clean up uploaded events set
    const validUploadedIds = new Set(data.events.map(e => e.id));
    data.uploadedEvents = data.uploadedEvents.filter(id => validUploadedIds.has(id));
    
    await this.setStorageData(data);
  }

  generateId() {
    return Date.now().toString(36) + Math.random().toString(36).substr(2);
  }

  extractDomain(url) {
    try {
      return new URL(url).hostname;
    } catch {
      return '';
    }
  }

  async getStats() {
    const data = await this.getStorageData();
    const recentContext = await this.getRecentContext();
    
    return {
      totalEvents: data.events.length,
      uploadedEvents: data.uploadedEvents.length,
      recentEvents: recentContext.events.length,
      modelVersion: data.modelVersion,
      storageSize: JSON.stringify(data).length,
      lastActivity: data.events.length > 0 ? data.events[0].timestamp : null
    };
  }
}