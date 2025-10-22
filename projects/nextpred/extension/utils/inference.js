/**
 * Inference Engine for Next Action Predictor
 * Handles ONNX model loading and inference
 */

export class InferenceEngine {
  constructor() {
    this.model = null;
    this.session = null;
    this.modelVersion = null;
    this.isLoaded = false;
    this.onnxRuntime = null;
  }

  async loadONNXRuntime() {
    if (this.onnxRuntime) return this.onnxRuntime;

    try {
      // Load ONNX Runtime WebAssembly
      const script = document.createElement('script');
      script.src = chrome.runtime.getURL('lib/onnxruntime-web.min.js');
      script.type = 'module';
      
      return new Promise((resolve, reject) => {
        script.onload = () => {
          this.onnxRuntime = window.ort;
          resolve(this.onnxRuntime);
        };
        script.onerror = reject;
        document.head.appendChild(script);
      });
    } catch (error) {
      console.error('Failed to load ONNX Runtime:', error);
      throw error;
    }
  }

  async loadModel() {
    try {
      if (this.isLoaded && this.session) {
        return true;
      }

      // Load ONNX Runtime
      await this.loadONNXRuntime();

      // Get model from storage
      const modelData = await this.getModelFromStorage();
      if (!modelData) {
        throw new Error('No model found in storage');
      }

      // Create inference session
      this.session = await this.onnxRuntime.InferenceSession.create(modelData, {
        executionProviders: ['wasm', 'cpu'],
        graphOptimizationLevel: 'all'
      });

      this.isLoaded = true;
      console.log('Model loaded successfully');
      return true;
    } catch (error) {
      console.error('Failed to load model:', error);
      this.isLoaded = false;
      return false;
    }
  }

  async getModelFromStorage() {
    try {
      const result = await chrome.storage.local.get('onnxModel');
      return result.onnxModel;
    } catch (error) {
      console.error('Failed to get model from storage:', error);
      return null;
    }
  }

  async downloadModel(version) {
    try {
      const { APIManager } = await import('./api.js');
      const api = new APIManager();

      console.log('Downloading model version:', version);
      const response = await api.downloadModel(version);

      if (response.success) {
        // Store model in local storage
        await chrome.storage.local.set({
          onnxModel: response.modelBlob,
          modelVersion: version,
          modelTimestamp: Date.now()
        });

        this.modelVersion = version;
        console.log('Model downloaded and stored successfully');
        return true;
      } else {
        throw new Error(response.error);
      }
    } catch (error) {
      console.error('Failed to download model:', error);
      return false;
    }
  }

  async predict(context, currentTab) {
    try {
      // If model is not loaded, use rule-based predictions
      if (!this.isLoaded) {
        return this.ruleBasedPredictions(context, currentTab);
      }

      // Prepare input features
      const features = this.prepareFeatures(context, currentTab);
      
      // Run inference
      const output = await this.runInference(features);
      
      // Post-process predictions
      const predictions = this.postProcessOutput(output, context, currentTab);
      
      return predictions;
    } catch (error) {
      console.error('Inference failed:', error);
      // Fallback to rule-based predictions
      return this.ruleBasedPredictions(context, currentTab);
    }
  }

  prepareFeatures(context, currentTab) {
    // Extract features from context and current tab
    const now = new Date();
    const features = {
      // Temporal features
      hour: now.getHours(),
      minute: now.getMinutes(),
      dayOfWeek: now.getDay(),
      isWeekend: now.getDay() === 0 || now.getDay() === 6,
      
      // Current state
      currentUrl: currentTab?.url || '',
      currentDomain: this.extractDomain(currentTab?.url || ''),
      currentTabIndex: currentTab?.index || 0,
      totalTabs: context.summary?.uniqueDomains?.length || 1,
      timeOnPage: this.calculateTimeOnPage(context, currentTab),
      scrollPosition: this.getAverageScrollPosition(context),
      
      // Recent history
      recentUrls: context.summary?.recentUrls || [],
      recentDomains: context.summary?.uniqueDomains || [],
      recentTabSwitches: this.getRecentTabSwitches(context),
      recentSearches: context.summary?.recentSearches || [],
      
      // Activity patterns
      tabSwitchFrequency: context.summary?.tabSwitches || 0,
      searchFrequency: context.summary?.searches || 0,
      scrollFrequency: context.summary?.scrolls || 0,
      
      // Time since last activity
      timeSinceLastTabSwitch: this.getTimeSinceLastActivity(context, 'tab_switch'),
      timeSinceLastSearch: this.getTimeSinceLastActivity(context, 'search'),
      timeSinceLastNavigation: this.getTimeSinceLastActivity(context, 'navigation')
    };

    return features;
  }

  async runInference(features) {
    try {
      // Convert features to tensor format expected by the model
      const inputTensor = this.featuresToTensor(features);
      
      // Run inference
      const results = await this.session.run({
        input: inputTensor
      });

      return results;
    } catch (error) {
      console.error('Inference execution failed:', error);
      throw error;
    }
  }

  featuresToTensor(features) {
    // This is a simplified version - in practice, you'd need proper tensor creation
    // based on your model's expected input format
    
    const featureVector = [
      features.hour / 24.0,
      features.minute / 60.0,
      features.dayOfWeek / 7.0,
      features.isWeekend ? 1.0 : 0.0,
      features.currentTabIndex / 10.0, // Normalize
      features.totalTabs / 20.0, // Normalize
      features.timeOnPage / 300.0, // Normalize (5 minutes)
      features.scrollPosition,
      features.tabSwitchFrequency / 10.0, // Normalize
      features.searchFrequency / 5.0, // Normalize
      features.scrollFrequency / 20.0, // Normalize
      features.timeSinceLastTabSwitch / 600.0, // Normalize (10 minutes)
      features.timeSinceLastSearch / 600.0,
      features.timeSinceLastNavigation / 600.0
    ];

    // Create Float32Array tensor
    const data = new Float32Array(featureVector);
    const tensor = new this.onnxRuntime.Tensor('float32', data, [1, featureVector.length]);
    
    return tensor;
  }

  postProcessOutput(output, context, currentTab) {
    // Extract predictions from model output
    // This is a simplified version - adapt based on your actual model output format
    
    const routerWeights = output.router?.data || [0.7, 0.2, 0.1]; // [tab, search, scroll]
    const tabPredictions = output.tab_expert?.data || [];
    const searchPrediction = output.search_expert?.data || '';
    const scrollPrediction = output.scroll_expert?.data || [0.5];

    const predictions = [];

    // Tab switch predictions
    if (routerWeights[0] > 0.3) { // If router suggests tab action
      const tabPreds = this.generateTabPredictions(context, currentTab, tabPredictions);
      predictions.push(...tabPreds);
    }

    // Search predictions
    if (routerWeights[1] > 0.3) { // If router suggests search action
      const searchPred = this.generateSearchPrediction(searchPrediction, currentTab);
      if (searchPred) {
        predictions.push(searchPred);
      }
    }

    // Scroll predictions (less common for tab switching)
    if (routerWeights[2] > 0.5 && predictions.length < 3) {
      const scrollPred = this.generateScrollPrediction(scrollPrediction[0], currentTab);
      if (scrollPred) {
        predictions.push(scrollPred);
      }
    }

    // Sort by confidence and return top 3
    return predictions
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 3);
  }

  ruleBasedPredictions(context, currentTab) {
    // Simple rule-based predictions when model is not available
    const predictions = [];
    const recentDomains = context.summary?.uniqueDomains || [];
    const recentUrls = context.summary?.recentUrls || [];

    // Predict returning to recently visited domains
    recentDomains.forEach((domain, index) => {
      if (domain && domain !== this.extractDomain(currentTab?.url || '')) {
        predictions.push({
          type: 'tab_switch',
          url: `https://${domain}`,
          title: domain,
          confidence: Math.max(0.1, 0.5 - index * 0.1),
          source: 'rule_based',
          domain
        });
      }
    });

    // Predict search based on recent searches
    if (context.summary?.recentSearches?.length > 0) {
      const lastSearch = context.summary.recentSearches[0];
      predictions.push({
        type: 'search',
        url: `https://www.google.com/search?q=${encodeURIComponent(lastSearch)}`,
        title: `Search: ${lastSearch}`,
        confidence: 0.3,
        source: 'rule_based',
        query: lastSearch
      });
    }

    return predictions
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 3);
  }

  generateTabPredictions(context, currentTab, modelPredictions) {
    const predictions = [];
    const recentUrls = context.summary?.recentUrls || [];

    // Use model predictions or fall back to recent URLs
    const candidateUrls = modelPredictions.length > 0 ? 
      modelPredictions : recentUrls.slice(0, 3);

    candidateUrls.forEach((url, index) => {
      if (url && url !== currentTab?.url) {
        predictions.push({
          type: 'tab_switch',
          url,
          title: this.extractTitle(url),
          confidence: Math.max(0.1, 0.8 - index * 0.2),
          source: 'model',
          tabId: null // Would need to map URL to actual tab ID
        });
      }
    });

    return predictions;
  }

  generateSearchPrediction(searchQuery, currentTab) {
    if (!searchQuery || searchQuery.trim().length === 0) return null;

    return {
      type: 'search',
      url: `https://www.google.com/search?q=${encodeURIComponent(searchQuery)}`,
      title: `Search: ${searchQuery}`,
      confidence: 0.6,
      source: 'model',
      query: searchQuery
    };
  }

  generateScrollPrediction(scrollPosition, currentTab) {
    // Scroll predictions are less common for tab switching
    // but could be useful for "scroll to top/bottom" actions
    return null; // Skip scroll predictions for now
  }

  // Helper methods
  extractDomain(url) {
    try {
      return new URL(url).hostname;
    } catch {
      return '';
    }
  }

  extractTitle(url) {
    try {
      const urlObj = new URL(url);
      return urlObj.hostname + urlObj.pathname;
    } catch {
      return url;
    }
  }

  calculateTimeOnPage(context, currentTab) {
    const pageLoadEvents = context.events.filter(e => 
      e.type === 'page_load' && e.data.url === currentTab?.url
    );

    if (pageLoadEvents.length === 0) return 0;

    const lastPageLoad = Math.max(...pageLoadEvents.map(e => e.timestamp));
    return Date.now() - lastPageLoad;
  }

  getAverageScrollPosition(context) {
    const scrollEvents = context.events.filter(e => e.type === 'scroll');
    if (scrollEvents.length === 0) return 0.5;

    const avgScroll = scrollEvents.reduce((sum, e) => 
      sum + (e.data.scrollPercentage || 0), 0) / scrollEvents.length;
    
    return avgScroll;
  }

  getRecentTabSwitches(context) {
    return context.events
      .filter(e => e.type === 'tab_switch')
      .map(e => e.data.tabIndex || 0)
      .slice(-5);
  }

  getTimeSinceLastActivity(context, activityType) {
    const activities = context.events.filter(e => e.type === activityType);
    if (activities.length === 0) return 10 * 60 * 1000; // 10 minutes

    const lastActivity = Math.max(...activities.map(e => e.timestamp));
    return Date.now() - lastActivity;
  }

  async getModelInfo() {
    return {
      loaded: this.isLoaded,
      version: this.modelVersion,
      timestamp: await this.getModelTimestamp()
    };
  }

  async getModelTimestamp() {
    try {
      const result = await chrome.storage.local.get('modelTimestamp');
      return result.modelTimestamp || null;
    } catch {
      return null;
    }
  }
}