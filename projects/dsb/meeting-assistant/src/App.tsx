import { useEffect, useCallback } from 'react';
import { useMeetingStore } from './store/meetingStore';
import { Subtitles } from './components/Subtitles';
import { SummaryPanel } from './components/SummaryPanel';
import { SettingsModal } from './components/SettingsModal';
import { MeetingEndSummary } from './components/MeetingEndSummary';
import './App.css';

function App() {
  const {
    isSessionActive,
    sessionStartTime,
    isLoading,
    error,
    showEndSummary,
    finalSummary,
    startSession,
    stopSession,
    toggleSettings,
    clearError,
    closeSummaryView,
    initializeListeners,
  } = useMeetingStore();

  // Initialize event listeners
  useEffect(() => {
    let cleanup: (() => void) | undefined;
    
    initializeListeners().then((unsubscribe) => {
      cleanup = unsubscribe;
    });

    return () => {
      if (cleanup) cleanup();
    };
  }, [initializeListeners]);

  // Keyboard shortcuts
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    // Ctrl/Cmd + Shift + S: Toggle settings
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'S') {
      e.preventDefault();
      toggleSettings();
    }
    // Escape: Close modals
    if (e.key === 'Escape') {
      if (showEndSummary) closeSummaryView();
    }
  }, [toggleSettings, showEndSummary, closeSummaryView]);

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  const handleSessionToggle = async () => {
    try {
      if (isSessionActive) {
        await stopSession();
      } else {
        await startSession();
      }
    } catch (err) {
      console.error('Session toggle error:', err);
    }
  };

  const formatSessionTime = () => {
    if (!sessionStartTime) return '00:00';
    const elapsed = Math.floor((Date.now() - sessionStartTime) / 1000);
    const minutes = Math.floor(elapsed / 60);
    const seconds = elapsed % 60;
    return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
  };

  return (
    <div className="app-container">
      {/* Control Bar */}
      <header className="control-bar">
        <div className="control-left">
          <h1 className="app-title">Meeting Assistant</h1>
          {isSessionActive && (
            <span className="session-timer">
              <span className="recording-dot" />
              {formatSessionTime()}
            </span>
          )}
        </div>
        
        <div className="control-center">
          <button
            className={`session-button ${isSessionActive ? 'active' : ''}`}
            onClick={handleSessionToggle}
            disabled={isLoading}
          >
            {isLoading ? (
              'Loading...'
            ) : isSessionActive ? (
              <>⏹ End Meeting</>
            ) : (
              <>▶ Start Meeting</>
            )}
          </button>
        </div>

        <div className="control-right">
          <button className="settings-button" onClick={toggleSettings}>
            ⚙️
          </button>
        </div>
      </header>

      {/* Error Banner */}
      {error && (
        <div className="error-banner">
          <span>{error}</span>
          <button onClick={clearError}>✕</button>
        </div>
      )}

      {/* Main Content */}
      <main className="main-content">
        <div className="content-area">
          {!isSessionActive && !showEndSummary && (
            <div className="welcome-message">
              <h2>Ready to assist your meeting</h2>
              <p>Click "Start Meeting" to begin real-time transcription and translation.</p>
              <p className="hint">Make sure to set your OpenAI API key in Settings first.</p>
            </div>
          )}
        </div>
        
        {isSessionActive && <SummaryPanel />}
      </main>

      {/* Subtitles */}
      {isSessionActive && <Subtitles />}

      {/* Settings Modal */}
      <SettingsModal />

      {/* End Summary Modal */}
      {showEndSummary && finalSummary && (
        <MeetingEndSummary
          summary={finalSummary}
          startTime={sessionStartTime}
          onClose={closeSummaryView}
        />
      )}
    </div>
  );
}

export default App;
