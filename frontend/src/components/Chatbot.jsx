import React, { useState } from 'react';
import axios from 'axios';
import { FaCommentDots, FaTimes, FaPaperPlane, FaUser, FaRobot } from 'react-icons/fa';
import './Chatbot.css';

// Set the API base URL to match your FastAPI server
const API_URL = 'http://127.0.0.1:5000/api';

const Chatbot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    { from: 'bot', text: 'Hello! I am the CyberGuard Bot. Ask me about this app or cybersecurity concepts.' }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const toggleChat = () => setIsOpen(!isOpen);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = { from: 'user', text: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await axios.post(`${API_URL}/chatbot`, { query: input });
      const botMessage = { from: 'bot', text: response.data.response };
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      const errorMessage = { from: 'bot', text: 'Sorry, I am having trouble connecting.' };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      <button className="chat-fab" onClick={toggleChat}>
        <FaCommentDots size={24} />
      </button>

      {isOpen && (
        <div className="chat-modal">
          <div className="chat-header">
            <h3>CyberGuard Bot</h3>
            <button onClick={toggleChat}><FaTimes /></button>
          </div>
          <div className="chat-body">
            {messages.map((msg, index) => (
              <div key={index} className={`chat-message ${msg.from}`}>
                <div className="chat-icon">
                  {msg.from === 'bot' ? <FaRobot /> : <FaUser />}
                </div>
                <div className="chat-bubble">{msg.text}</div>
              </div>
            ))}
            {isLoading && (
              <div className="chat-message bot">
                <div className="chat-icon"><FaRobot /></div>
                <div className="chat-bubble typing-indicator">
                  <span></span><span></span><span></span>
                </div>
              </div>
            )}
          </div>
          <form className="chat-input" onSubmit={handleSubmit}>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question..."
              disabled={isLoading}
            />
            <button type="submit" disabled={isLoading}><FaPaperPlane /></button>
          </form>
        </div>
      )}
    </>
  );
};

export default Chatbot;