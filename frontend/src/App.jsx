import React from 'react';
import Header from './components/Header';
import Dashboard from './components/Dashboard.jsx';
import Chatbot from './components/Chatbot';

function App() {
  return (
    <div className="App">
      <Header />
      <main className="container">
        <Dashboard />
      </main>
      <Chatbot />
    </div>
  );
}

export default App;