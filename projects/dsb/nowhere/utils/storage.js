/**
 * Storage utility for managing Chrome extension storage
 */
const Storage = {
  /**
   * Save API key to chrome.storage.local
   * @param {string} apiKey - The API key to save
   * @returns {Promise<void>}
   */
  setApiKey: (apiKey) => {
    return new Promise((resolve, reject) => {
      chrome.storage.local.set({ apiKey }, () => {
        if (chrome.runtime.lastError) {
          reject(new Error(chrome.runtime.lastError.message));
        } else {
          resolve();
        }
      });
    });
  },

  /**
   * Get API key from chrome.storage.local
   * @returns {Promise<string|null>} The stored API key or null if not found
   */
  getApiKey: () => {
    return new Promise((resolve, reject) => {
      chrome.storage.local.get('apiKey', (result) => {
        if (chrome.runtime.lastError) {
          reject(new Error(chrome.runtime.lastError.message));
        } else {
          resolve(result.apiKey || null);
        }
      });
    });
  },

  /**
   * Remove API key from chrome.storage.local
   * @returns {Promise<void>}
   */
  removeApiKey: () => {
    return new Promise((resolve, reject) => {
      chrome.storage.local.remove('apiKey', () => {
        if (chrome.runtime.lastError) {
          reject(new Error(chrome.runtime.lastError.message));
        } else {
          resolve();
        }
      });
    });
  }
};