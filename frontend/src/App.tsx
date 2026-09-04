import React, { useState, useRef, useEffect } from 'react';
import { ChatMessage, type MessageProps } from './components/ChatMessage';
import { ChatInput } from './components/ChatInput';
import { BrainCircuit } from 'lucide-react';

interface ChatMessageState extends MessageProps {
  id: string;
}

const App: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessageState[]>([{
    id: 'welcome',
    role: 'ai',
    content: "Hello! I am Synapse, a neuro-symbolic agentic RAG assistant. How can I help you today?"
  }]);
  const [isLoading, setIsLoading] = useState(false);
  const endOfMessagesRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSend = async (text: string) => {
    // Add user message
    const userMsg: ChatMessageState = {
      id: Date.now().toString(),
      role: 'user',
      content: text
    };
    
    setMessages(prev => [...prev, userMsg]);
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query: text,
          include_trace: true
        })
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      
      const aiMsg: ChatMessageState = {
        id: (Date.now() + 1).toString(),
        role: 'ai',
        content: data.answer || "Sorry, I couldn't generate an answer.",
        evidenceConfidence: data.evidence_confidence,
        citations: data.citations,
        retrievalStrategy: data.retrieval_strategy
      };

      setMessages(prev => [...prev, aiMsg]);
    } catch (error) {
      console.error('Error fetching from API:', error);
      const errorMsg: ChatMessageState = {
        id: (Date.now() + 1).toString(),
        role: 'ai',
        content: "Sorry, I encountered an error communicating with the backend server. Is it running?"
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
          <BrainCircuit color="#3b82f6" size={28} />
          <h1>Synapse</h1>
        </div>
        <p>Evidence-Grounded Question Answering</p>
      </header>

      <main className="chat-container">
        {messages.map(msg => (
          <ChatMessage 
            key={msg.id} 
            {...msg} 
          />
        ))}
        {isLoading && (
          <div className="message-wrapper ai">
            <div className="message ai">
              <div className="typing-indicator">
                <div className="typing-dot"></div>
                <div className="typing-dot"></div>
                <div className="typing-dot"></div>
              </div>
            </div>
          </div>
        )}
        <div ref={endOfMessagesRef} />
      </main>

      <ChatInput onSend={handleSend} disabled={isLoading} />
    </div>
  );
};

export default App;
