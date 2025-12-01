import React from 'react';
import { useMeetingStore } from '../store/meetingStore';
import './SummaryPanel.css';

export const SummaryPanel: React.FC = () => {
  const { summary, checkKeywordMatch } = useMeetingStore();

  return (
    <div className="summary-panel">
      <div className="summary-section">
        <h3 className="section-title">📋 Summary</h3>
        <ul className="section-list">
          {summary.summary.length === 0 ? (
            <li className="empty-item">No summary yet...</li>
          ) : (
            summary.summary.map((item, index) => (
              <li 
                key={index} 
                className={`list-item ${checkKeywordMatch(item) ? 'highlighted' : ''}`}
              >
                {item}
              </li>
            ))
          )}
        </ul>
      </div>

      <div className="summary-section">
        <h3 className="section-title">✅ Decisions</h3>
        <ul className="section-list">
          {summary.decisions.length === 0 ? (
            <li className="empty-item">No decisions yet...</li>
          ) : (
            summary.decisions.map((item, index) => (
              <li 
                key={index}
                className={`list-item ${checkKeywordMatch(item) ? 'highlighted' : ''}`}
              >
                {item}
              </li>
            ))
          )}
        </ul>
      </div>

      <div className="summary-section">
        <h3 className="section-title">📌 Action Items</h3>
        <ul className="section-list">
          {summary.action_items.length === 0 ? (
            <li className="empty-item">No action items yet...</li>
          ) : (
            summary.action_items.map((item, index) => (
              <li 
                key={index}
                className={`list-item ${item.isRelevantToUser || checkKeywordMatch(item.text) ? 'highlighted user-relevant' : ''}`}
              >
                {item.isRelevantToUser && <span className="user-badge">👤</span>}
                {item.text}
              </li>
            ))
          )}
        </ul>
      </div>
    </div>
  );
};
