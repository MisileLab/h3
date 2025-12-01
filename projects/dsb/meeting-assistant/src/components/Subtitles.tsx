import React from 'react';
import { useMeetingStore } from '../store/meetingStore';
import './Subtitles.css';

export const Subtitles: React.FC = () => {
  const { currentSubtitle, subtitles, checkKeywordMatch } = useMeetingStore();
  
  // Get the last few subtitles for context
  const recentSubtitles = subtitles.slice(-3);
  
  if (!currentSubtitle && recentSubtitles.length === 0) {
    return (
      <div className="subtitles-container">
        <div className="subtitles-placeholder">
          Waiting for speech...
        </div>
      </div>
    );
  }

  const displaySubtitle = currentSubtitle || recentSubtitles[recentSubtitles.length - 1];
  const isHighlighted = displaySubtitle && (
    checkKeywordMatch(displaySubtitle.text_en) || 
    checkKeywordMatch(displaySubtitle.text_ko)
  );

  return (
    <div className="subtitles-container">
      {displaySubtitle && (
        <div className={`subtitle-block ${isHighlighted ? 'highlighted' : ''}`}>
          <div className="subtitle-korean">
            {displaySubtitle.text_ko || '번역 중...'}
          </div>
          <div className="subtitle-english">
            {displaySubtitle.text_en}
          </div>
          {displaySubtitle.speaker && (
            <div className="subtitle-speaker">
              {displaySubtitle.speaker}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
