import React, { useState, useEffect } from 'react';
import { useMeetingStore } from '../store/meetingStore';
import './SettingsModal.css';

export const SettingsModal: React.FC = () => {
  const { settings, isSettingsOpen, toggleSettings, setApiKey, setKeywords, updateSettings } = useMeetingStore();
  
  const [localUsername, setLocalUsername] = useState(settings.username);
  const [localKeywords, setLocalKeywords] = useState(settings.keywords.join(', '));
  const [localApiKey, setLocalApiKey] = useState(settings.apiKey);
  const [isSaving, setIsSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  useEffect(() => {
    if (isSettingsOpen) {
      setLocalUsername(settings.username);
      setLocalKeywords(settings.keywords.join(', '));
      setLocalApiKey(settings.apiKey);
      setSaveError(null);
    }
  }, [isSettingsOpen, settings]);

  if (!isSettingsOpen) return null;

  const handleSave = async () => {
    setIsSaving(true);
    setSaveError(null);

    try {
      // Update username
      updateSettings({ username: localUsername });

      // Parse and set keywords
      const keywordsArray = localKeywords
        .split(',')
        .map(k => k.trim())
        .filter(k => k.length > 0);
      await setKeywords(keywordsArray);

      // Set API key if changed
      if (localApiKey && localApiKey !== settings.apiKey) {
        await setApiKey(localApiKey);
      }

      toggleSettings();
    } catch (error) {
      setSaveError(String(error));
    } finally {
      setIsSaving(false);
    }
  };

  const handleOverlayClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      toggleSettings();
    }
  };

  return (
    <div className="settings-overlay" onClick={handleOverlayClick}>
      <div className="settings-modal">
        <div className="settings-header">
          <h2>Settings</h2>
          <button className="close-button" onClick={toggleSettings}>✕</button>
        </div>

        <div className="settings-content">
          <div className="settings-field">
            <label htmlFor="username">Your Name</label>
            <input
              id="username"
              type="text"
              value={localUsername}
              onChange={(e) => setLocalUsername(e.target.value)}
              placeholder="Enter your name"
            />
            <span className="field-hint">Used to identify mentions in the meeting</span>
          </div>

          <div className="settings-field">
            <label htmlFor="keywords">Keywords to Watch</label>
            <textarea
              id="keywords"
              value={localKeywords}
              onChange={(e) => setLocalKeywords(e.target.value)}
              placeholder="project-x, team-alpha, budget review"
              rows={3}
            />
            <span className="field-hint">Comma-separated list of keywords to highlight</span>
          </div>

          <div className="settings-field">
            <label htmlFor="apiKey">OpenAI API Key</label>
            <input
              id="apiKey"
              type="password"
              value={localApiKey}
              onChange={(e) => setLocalApiKey(e.target.value)}
              placeholder="sk-..."
            />
            <span className="field-hint">Your API key is stored locally and never shared</span>
          </div>

          {saveError && (
            <div className="settings-error">
              {saveError}
            </div>
          )}
        </div>

        <div className="settings-footer">
          <button className="cancel-button" onClick={toggleSettings}>
            Cancel
          </button>
          <button 
            className="save-button" 
            onClick={handleSave}
            disabled={isSaving}
          >
            {isSaving ? 'Saving...' : 'Save Settings'}
          </button>
        </div>
      </div>
    </div>
  );
};
