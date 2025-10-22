/**
 * Content Script for Next Action Predictor
 * Tracks scroll positions and user interactions on web pages
 */

class ContentTracker {
  constructor() {
    this.scrollThreshold = 100; // pixels
    this.lastScrollPosition = 0;
    this.scrollStartTime = Date.now();
    this.isTracking = true;
    
    this.init();
  }

  init() {
    // Don't track on sensitive pages
    if (this.isSensitivePage()) {
      return;
    }

    this.setupScrollTracking();
    this.setupMessageListener();
    this.setupClickTracking();
    this.setupSearchTracking();
  }

  setupScrollTracking() {
    let scrollTimeout;
    
    window.addEventListener('scroll', () => {
      if (!this.isTracking) return;

      clearTimeout(scrollTimeout);
      scrollTimeout = setTimeout(() => {
        this.handleScroll();
      }, 100);
    });

    // Track initial scroll position
    setTimeout(() => {
      this.handleScroll();
    }, 1000);
  }

  handleScroll() {
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
    const scrollHeight = document.documentElement.scrollHeight - window.innerHeight;
    const scrollPercentage = scrollHeight > 0 ? scrollTop / scrollHeight : 0;

    // Only track significant scroll changes
    if (Math.abs(scrollTop - this.lastScrollPosition) > this.scrollThreshold) {
      this.recordScrollEvent(scrollTop, scrollPercentage);
      this.lastScrollPosition = scrollTop;
    }
  }

  recordScrollEvent(scrollTop, scrollPercentage) {
    const event = {
      type: 'scroll',
      timestamp: Date.now(),
      data: {
        url: window.location.href,
        scrollTop,
        scrollPercentage,
        pageHeight: document.documentElement.scrollHeight,
        viewportHeight: window.innerHeight,
        timeOnPage: Date.now() - this.scrollStartTime
      }
    };

    // Send to background script
    chrome.runtime.sendMessage({
      action: 'recordEvent',
      event
    });
  }

  setupClickTracking() {
    document.addEventListener('click', (event) => {
      if (!this.isTracking) return;

      const target = event.target;
      const linkElement = target.closest('a');
      
      if (linkElement && linkElement.href) {
        this.recordClickEvent(linkElement.href, linkElement.textContent, target);
      }
    });
  }

  recordClickEvent(href, text, element) {
    const event = {
      type: 'click',
      timestamp: Date.now(),
      data: {
        url: window.location.href,
        targetUrl: href,
        linkText: text?.trim() || '',
        elementTag: element.tagName,
        elementClass: element.className,
        position: {
          x: element.getBoundingClientRect().left,
          y: element.getBoundingClientRect().top
        }
      }
    };

    chrome.runtime.sendMessage({
      action: 'recordEvent',
      event
    });
  }

  setupSearchTracking() {
    // Track search form submissions
    document.addEventListener('submit', (event) => {
      const form = event.target;
      const searchInput = form.querySelector('input[type="search"], input[name="q"], input[name="query"]');
      
      if (searchInput && searchInput.value) {
        this.recordSearchEvent(searchInput.value, form.action);
      }
    });

    // Track search input changes (with debounce)
    let searchTimeout;
    document.addEventListener('input', (event) => {
      const target = event.target;
      if (target.type === 'search' || target.name === 'q' || target.name === 'query') {
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(() => {
          if (target.value && target.value.length > 2) {
            this.recordSearchEvent(target.value, window.location.href, 'input');
          }
        }, 500);
      }
    });
  }

  recordSearchEvent(query, url, source = 'form') {
    const event = {
      type: 'search',
      timestamp: Date.now(),
      data: {
        url: window.location.href,
        searchUrl: url,
        query: query.trim(),
        source,
        isGoogleSearch: window.location.hostname.includes('google.com'),
        isInternalSearch: url.includes(window.location.hostname)
      }
    };

    chrome.runtime.sendMessage({
      action: 'recordEvent',
      event
    });
  }

  setupMessageListener() {
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
      if (request.action === 'getScrollPosition') {
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        const scrollHeight = document.documentElement.scrollHeight - window.innerHeight;
        const scrollPercentage = scrollHeight > 0 ? scrollTop / scrollHeight : 0;

        sendResponse({
          position: scrollPercentage,
          scrollTop,
          height: document.documentElement.scrollHeight
        });
      }

      if (request.action === 'pauseTracking') {
        this.isTracking = false;
      }

      if (request.action === 'resumeTracking') {
        this.isTracking = true;
      }
    });
  }

  isSensitivePage() {
    const url = window.location.href;
    const sensitivePatterns = [
      /accounts\./,
      /auth\./,
      /login/,
      /register/,
      /signup/,
      /token=/,
      /session=/,
      /key=/,
      /password/,
      /bank/,
      /payment/,
      /checkout/,
      /cart/
    ];

    return sensitivePatterns.some(pattern => pattern.test(url));
  }
}

// Initialize content tracker
const contentTracker = new ContentTracker();