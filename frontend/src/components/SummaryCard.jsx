import React from 'react';
import './SummaryCard.css';

const SummaryCard = ({ icon, title, value, color }) => {
  return (
    <div className="summary-card">
      <div className="card-icon" style={{ backgroundColor: color }}>
        {icon}
      </div>
      <div className="card-content">
        <h3 className="card-title">{title}</h3>
        <p className="card-value">{value}</p>
      </div>
    </div>
  );
};

export default SummaryCard;