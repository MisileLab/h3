/**
 * API Manager for Next Action Predictor
 * Handles communication with the backend server
 */

export class APIManager {
  constructor() {
    this.baseUrl = this.getBaseUrl();
    this.timeout = 10000; // 10 seconds
  }

  getBaseUrl() {
    // In development, use localhost
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
      return 'http://localhost:8000';
    }
    
    // In production, use configured server URL
    return 'https://your-nextpred-server.com';
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseUrl}${endpoint}`;
    const config = {
      timeout: this.timeout,
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': 'NextActionPredictor/1.0',
        ...options.headers
      },
      ...options
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error(`API request failed: ${endpoint}`, error);
      throw error;
    }
  }

  async uploadEvents(events) {
    try {
      const payload = {
        events: events.map(event => this.sanitizeEvent(event)),
        metadata: {
          userAgent: navigator.userAgent,
          timestamp: Date.now(),
          version: chrome.runtime.getManifest().version
        }
      };

      const response = await this.request('/api/events/batch', {
        method: 'POST',
        body: JSON.stringify(payload)
      });

      return { success: true, uploaded: events.length, response };
    } catch (error) {
      console.error('Failed to upload events:', error);
      return { success: false, error: error.message };
    }
  }

  async getModelVersion() {
    try {
      const response = await this.request('/api/model/version');
      return response;
    } catch (error) {
      console.error('Failed to get model version:', error);
      return { version: null, timestamp: null };
    }
  }

  async downloadModel(version) {
    try {
      const response = await this.request(`/api/model/download?version=${version}`);
      
      if (response.modelUrl) {
        // Download the model file
        const modelResponse = await fetch(response.modelUrl);
        const modelBlob = await modelResponse.blob();
        
        return {
          success: true,
          modelBlob,
          version: response.version,
          checksum: response.checksum
        };
      }
      
      throw new Error('No model URL provided');
    } catch (error) {
      console.error('Failed to download model:', error);
      return { success: false, error: error.message };
    }
  }

  async recordFeedback(feedback) {
    try {
      const response = await this.request('/api/feedback', {
        method: 'POST',
        body: JSON.stringify({
          feedback: this.sanitizeFeedback(feedback),
          metadata: {
            timestamp: Date.now(),
            userAgent: navigator.userAgent,
            version: chrome.runtime.getManifest().version
          }
        })
      });

      return { success: true, response };
    } catch (error) {
      console.error('Failed to record feedback:', error);
      return { success: false, error: error.message };
    }
  }

  async getUserStats() {
    try {
      const response = await this.request('/api/user/stats');
      return response;
    } catch (error) {
      console.error('Failed to get user stats:', error);
      return { success: false, error: error.message };
    }
  }

  async checkServerHealth() {
    try {
      const response = await this.request('/api/health');
      return { 
        healthy: true, 
        server: response.server,
        version: response.version,
        timestamp: response.timestamp
      };
    } catch (error) {
      return { healthy: false, error: error.message };
    }
  }

  sanitizeEvent(event) {
    // Remove sensitive information from events
    const sanitized = { ...event };

    if (sanitized.data && sanitized.data.url) {
      sanitized.data.url = this.sanitizeUrl(sanitized.data.url);
    }

    if (sanitized.data && sanitized.data.targetUrl) {
      sanitized.data.targetUrl = this.sanitizeUrl(sanitized.data.targetUrl);
    }

    // Remove any potential PII
    if (sanitized.data && sanitized.data.query) {
      sanitized.data.query = this.sanitizeSearchQuery(sanitized.data.query);
    }

    return sanitized;
  }

  sanitizeUrl(url) {
    try {
      const urlObj = new URL(url);
      
      // Remove sensitive query parameters
      const sensitiveParams = [
        'token', 'session', 'key', 'password', 'auth', 'access_token',
        'refresh_token', 'api_key', 'secret', 'csrf', 'ssid'
      ];

      const params = new URLSearchParams(urlObj.search);
      sensitiveParams.forEach(param => {
        params.delete(param);
      });

      urlObj.search = params.toString();
      return urlObj.toString();
    } catch {
      return url;
    }
  }

  sanitizeSearchQuery(query) {
    // Remove potential sensitive information from search queries
    const sensitivePatterns = [
      /\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b/, // Credit card numbers
      /\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b/, // SSN patterns
      /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/ // Email addresses
    ];

    let sanitized = query;
    sensitivePatterns.forEach(pattern => {
      sanitized = sanitized.replace(pattern, '[REDACTED]');
    });

    return sanitized;
  }

  sanitizeFeedback(feedback) {
    const sanitized = { ...feedback };

    // Remove any URLs from feedback content
    if (sanitized.content) {
      sanitized.content = sanitized.content.replace(/https?:\/\/[^\s]+/g, '[URL]');
    }

    return sanitized;
  }

  async isOnline() {
    try {
      const response = await fetch(`${this.baseUrl}/api/health`, {
        method: 'HEAD',
        cache: 'no-cache'
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  async retryRequest(fn, maxRetries = 3, delay = 1000) {
    for (let i = 0; i < maxRetries; i++) {
      try {
        return await fn();
      } catch (error) {
        if (i === maxRetries - 1) throw error;
        
        // Exponential backoff
        await new Promise(resolve => setTimeout(resolve, delay * Math.pow(2, i)));
      }
    }
  }
}