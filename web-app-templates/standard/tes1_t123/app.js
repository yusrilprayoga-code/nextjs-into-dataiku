/* Javascript code for your Webapp tes1_t123
 *
 * This file handles the integration between the static Next.js frontend
 * and the Dataiku backend API endpoints.
 *
 * The static files are served from /resources/out/ and API calls are
 * proxied to the Python backend using getWebAppBackendUrl()
 */

// Global configuration
window.DATAIKU_CONFIG = {
  // API base URL will be set by the HTML file
  apiBaseUrl: null,

  // Initialize the configuration
  init: function() {
    // Set up API proxy
    this.setupApiProxy();

    // Set up error handling
    this.setupErrorHandling();

    console.log('Dataiku integration initialized');
  },

  // Set up API proxy to route calls to Dataiku backend
  setupApiProxy: function() {
    var originalFetch = window.fetch;

    window.fetch = function(url, options) {
      // Check if this is an API call that needs to be proxied
      if (typeof url === 'string' && url.includes('/api/')) {
        // Convert to Dataiku backend URL
        var backendUrl;
        if (typeof window.getWebAppBackendUrl === 'function') {
          var endpoint = url.replace('/api/', '').replace(/^\//, '');
          backendUrl = window.getWebAppBackendUrl(endpoint);
        } else {
          // Fallback for development
          backendUrl = url;
        }

        console.log('Proxying API call:', url, '->', backendUrl);

        // Add credentials for Dataiku
        options = options || {};
        options.credentials = 'same-origin';

        return originalFetch(backendUrl, options);
      }

      return originalFetch(url, options);
    };
  },

  // Set up global error handling
  setupErrorHandling: function() {
    window.addEventListener('error', function(e) {
      console.error('Global error:', e.error);
      // You can send error reports to Dataiku backend here if needed
    });

    window.addEventListener('unhandledrejection', function(e) {
      console.error('Unhandled promise rejection:', e.reason);
      // You can send error reports to Dataiku backend here if needed
    });
  },

  // Utility function to get Dataiku dataset data
  getDatasetData: function(datasetName, options) {
    options = options || {};
    var maxRows = options.maxRows || 500;
    var sampling = options.sampling || 'head';

    return fetch('/api/dataset-data?dataset=' + encodeURIComponent(datasetName) +
                 '&max_rows=' + maxRows + '&sampling=' + sampling)
      .then(function(response) {
        if (!response.ok) {
          throw new Error('Failed to fetch dataset: ' + response.statusText);
        }
        return response.json();
      });
  },

  // Utility function to show Dataiku notifications
  showNotification: function(message, type) {
    type = type || 'info';

    // Use Dataiku's notification system if available
    if (typeof window.dataiku && window.dataiku.notification) {
      window.dataiku.notification.show(message, type);
    } else {
      // Fallback to console
      console.log('[' + type.toUpperCase() + ']', message);

      // You could also show a simple alert or create a custom notification
      if (type === 'error') {
        alert('Error: ' + message);
      }
    }
  }
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
  window.DATAIKU_CONFIG.init();
});

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
  module.exports = window.DATAIKU_CONFIG;
}