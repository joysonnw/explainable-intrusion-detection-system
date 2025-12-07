import React from 'react';
import { FaShieldAlt } from 'react-icons/fa';
import './Header.css';

const Header = () => {
  return (
    <header className="app-header">
      <div className="container header-content">
        <div className="logo">
          <FaShieldAlt size={32} color="var(--color-accent)" />
          <h1>Intrusion Detection System</h1>
        </div>
        <nav>
          <span>Status: <span className="status-dot"></span>Online</span>
        </nav>
      </div>
    </header>
  );
};

export default Header;