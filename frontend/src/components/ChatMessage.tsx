import React from 'react';
import ReactMarkdown from 'react-markdown';
import { User, Sparkles } from 'lucide-react';

interface Citation {
  id: string;
  source: string;
  content: string;
}

export interface MessageProps {
  role: 'user' | 'ai';
  content: string;
  evidenceConfidence?: number;
  citations?: Citation[];
  retrievalStrategy?: string;
}

export const ChatMessage: React.FC<MessageProps> = ({ role, content, evidenceConfidence, citations, retrievalStrategy }) => {
  const isUser = role === 'user';
  
  return (
    <div className={`message-wrapper ${isUser ? 'user' : 'ai'}`}>
      {!isUser && (
        <div style={{ marginRight: '12px', marginTop: '4px' }}>
          <div style={{ background: 'var(--accent-color)', padding: '6px', borderRadius: '50%', color: 'white' }}>
            <Sparkles size={16} />
          </div>
        </div>
      )}
      
      <div className={`message ${isUser ? 'user' : 'ai'}`}>
        <div className="message-content">
          {isUser ? (
            <p>{content}</p>
          ) : (
            <ReactMarkdown>{content}</ReactMarkdown>
          )}
        </div>
        
        {!isUser && (evidenceConfidence !== undefined || (citations && citations.length > 0)) && (
          <div className="message-meta">
            {evidenceConfidence !== undefined && (
              <div className={`confidence-badge ${evidenceConfidence < 0.6 ? 'low' : ''}`}>
                Grounding Confidence: {Math.round(evidenceConfidence * 100)}%
              </div>
            )}
            
            {retrievalStrategy && (
              <div style={{ marginBottom: '0.5rem', color: '#94a3b8' }}>
                Strategy: <code>{retrievalStrategy}</code>
              </div>
            )}

            {citations && citations.length > 0 && (
              <div className="citation-list">
                <div style={{ fontWeight: 600, color: '#cbd5e1', marginBottom: '4px' }}>Sources:</div>
                {citations.map((c, i) => (
                  <div key={i} className="citation-item">
                    <div className="citation-title">[{c.id}] {c.source}</div>
                    <div style={{ fontSize: '0.85em', color: '#94a3b8' }}>
                      {c.content.length > 150 ? c.content.substring(0, 150) + '...' : c.content}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
      
      {isUser && (
        <div style={{ marginLeft: '12px', marginTop: '4px' }}>
          <div style={{ background: '#334155', padding: '6px', borderRadius: '50%', color: 'white' }}>
            <User size={16} />
          </div>
        </div>
      )}
    </div>
  );
};
