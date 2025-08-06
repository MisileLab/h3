/**
 * API utility for interacting with YouTube Multimodal Search API
 */
const Api = {
  // Base API URL - replace with actual API URL in production
  BASE_URL: 'https://api.youtubesearch.example',

  /**
   * Upload a video to the API
   * @param {string} videoId - YouTube video ID
   * @param {string} apiKey - API key for authentication
   * @returns {Promise<Object>} - API response
   */
  uploadVideo: async (videoId, apiKey) => {
    try {
      const response = await fetch(`${Api.BASE_URL}/api/videos/upload`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': apiKey
        },
        body: JSON.stringify({ videoId })
      });

      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.detail || 'Failed to upload video');
      }
      
      return data;
    } catch (error) {
      console.error('Upload error:', error);
      throw error;
    }
  },

  /**
   * Check embedding status for videos
   * @param {string[]} videoIds - Array of YouTube video IDs
   * @param {string} apiKey - API key for authentication
   * @returns {Promise<Object>} - Status information
   */
  checkEmbeddingStatus: async (videoIds, apiKey) => {
    try {
      const queryParams = new URLSearchParams();
      videoIds.forEach(id => queryParams.append('videoIds', id));
      
      const response = await fetch(`${Api.BASE_URL}/api/embeddings/ready?${queryParams}`, {
        method: 'GET',
        headers: {
          'X-API-Key': apiKey
        }
      });

      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.detail || 'Failed to check embedding status');
      }
      
      // Process the response to determine ready count
      const readyCount = data.filter(item => item.ready).length;
      const allReady = readyCount === videoIds.length;
      
      return {
        data,
        readyCount,
        allReady
      };
    } catch (error) {
      console.error('Status check error:', error);
      throw error;
    }
  },

  /**
   * Search videos using natural language query
   * @param {string} query - Search query
   * @param {string} apiKey - API key for authentication
   * @returns {Promise<Array>} - Search results
   */
  searchVideos: async (query, apiKey) => {
    try {
      const response = await fetch(`${Api.BASE_URL}/api/search/text`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': apiKey
        },
        body: JSON.stringify({ query })
      });

      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.detail || 'Search failed');
      }
      
      return data.results || [];
    } catch (error) {
      console.error('Search error:', error);
      throw error;
    }
  }
};