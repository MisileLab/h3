import React from 'react';
import { useMeetingStore } from '../store/meetingStore';
import type { MeetingSummary } from '../types';
import './MeetingEndSummary.css';

interface Props {
  summary: MeetingSummary;
  startTime: number | null;
  onClose: () => void;
}

export const MeetingEndSummary: React.FC<Props> = ({ summary, startTime, onClose }) => {
  const { checkKeywordMatch } = useMeetingStore();

  const formatDuration = () => {
    if (!startTime) return 'Unknown duration';
    const duration = Date.now() - startTime;
    const minutes = Math.floor(duration / 60000);
    const seconds = Math.floor((duration % 60000) / 1000);
    return `${minutes}m ${seconds}s`;
  };

  const formatTime = (timestamp: number | null) => {
    if (!timestamp) return 'N/A';
    return new Date(timestamp).toLocaleTimeString();
  };

  const handleCopy = () => {
    const text = generateTextSummary();
    navigator.clipboard.writeText(text);
  };

  const generateTextSummary = () => {
    const lines: string[] = [];
    lines.push('=== Meeting Summary ===');
    lines.push(`Duration: ${formatDuration()}`);
    lines.push(`Started: ${formatTime(startTime)}`);
    lines.push('');
    
    lines.push('## Key Points');
    summary.summary.forEach(item => lines.push(`- ${item}`));
    lines.push('');
    
    lines.push('## Decisions');
    summary.decisions.forEach(item => lines.push(`- ${item}`));
    lines.push('');
    
    lines.push('## Action Items');
    // User's action items first
    const userItems = summary.action_items.filter(item => 
      item.isRelevantToUser || checkKeywordMatch(item.text)
    );
    const otherItems = summary.action_items.filter(item => 
      !item.isRelevantToUser && !checkKeywordMatch(item.text)
    );
    
    if (userItems.length > 0) {
      lines.push('### My Actions');
      userItems.forEach(item => lines.push(`- ${item.text}`));
    }
    if (otherItems.length > 0) {
      lines.push('### Other Actions');
      otherItems.forEach(item => lines.push(`- ${item.text}`));
    }
    
    return lines.join('\n');
  };

  // Sort action items - user-relevant first
  const sortedActionItems = [...summary.action_items].sort((a, b) => {
    const aRelevant = a.isRelevantToUser || checkKeywordMatch(a.text);
    const bRelevant = b.isRelevantToUser || checkKeywordMatch(b.text);
    if (aRelevant && !bRelevant) return -1;
    if (!aRelevant && bRelevant) return 1;
    return 0;
  });

  return (
    <div className="end-summary-overlay">
      <div className="end-summary-modal">
        <div className="end-summary-header">
          <div className="header-info">
            <h2>Meeting Summary</h2>
            <div className="meeting-meta">
              <span className="meta-item">
                ⏱️ Duration: {formatDuration()}
              </span>
              <span className="meta-item">
                🕐 Started: {formatTime(startTime)}
              </span>
            </div>
          </div>
          <div className="header-actions">
            <button className="copy-button" onClick={handleCopy}>
              📋 Copy
            </button>
            <button className="close-button" onClick={onClose}>
              ✕
            </button>
          </div>
        </div>

        <div className="end-summary-content">
          <section className="summary-section">
            <h3>📋 Key Points</h3>
            {summary.summary.length === 0 ? (
              <p className="empty-message">No key points captured</p>
            ) : (
              <ul>
                {summary.summary.map((item, index) => (
                  <li 
                    key={index}
                    className={checkKeywordMatch(item) ? 'highlighted' : ''}
                  >
                    {item}
                  </li>
                ))}
              </ul>
            )}
          </section>

          <section className="summary-section">
            <h3>✅ Decisions</h3>
            {summary.decisions.length === 0 ? (
              <p className="empty-message">No decisions captured</p>
            ) : (
              <ul>
                {summary.decisions.map((item, index) => (
                  <li 
                    key={index}
                    className={checkKeywordMatch(item) ? 'highlighted' : ''}
                  >
                    {item}
                  </li>
                ))}
              </ul>
            )}
          </section>

          <section className="summary-section">
            <h3>📌 Action Items</h3>
            {summary.action_items.length === 0 ? (
              <p className="empty-message">No action items captured</p>
            ) : (
              <ul>
                {sortedActionItems.map((item, index) => {
                  const isUserItem = item.isRelevantToUser || checkKeywordMatch(item.text);
                  return (
                    <li 
                      key={index}
                      className={isUserItem ? 'highlighted user-item' : ''}
                    >
                      {isUserItem && <span className="user-badge">👤 MY ACTION</span>}
                      {item.text}
                    </li>
                  );
                })}
              </ul>
            )}
          </section>
        </div>
      </div>
    </div>
  );
};
