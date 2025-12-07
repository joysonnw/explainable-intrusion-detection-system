import React from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell        // ← IMPORTANT
} from 'recharts'
import './Analysis.css'

// SAME color scheme you used before
const COLORS = [
  '#10B981', // Benign
  '#EF4444', // Danger
  '#F59E0B', // PortScan (yellow)
  '#3B82F6', // Accent
  '#6366F1', // Indigo
  '#EC4899', // Pink
  '#8B5CF6'  // Violet
]

const Analysis = ({ data }) => {
  if (!data || data.length === 0) {
    return (
      <div className="analysis-container">
        <h3>Threat Analysis</h3>
        <div className="placeholder-text">
          Upload a file to see threat analysis.
        </div>
      </div>
    )
  }

  const chartData = data.map(item => ({
    name: item.name,
    value: item.value,
    percent: Number(item.percent).toFixed(2)
  }))

  return (
    <div className="analysis-container">
      <h3>Threat Analysis</h3>

      <div className="chart-wrapper">
        <ResponsiveContainer width="100%" height={350}>
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 10, right: 30, left: 40, bottom: 10 }}
          >
            <XAxis type="number" />
            <YAxis dataKey="name" type="category" width={120} />

            <Tooltip
              formatter={(value, name, props) => [
                `${props.payload.percent}% (${value} rows)`,
                'Count'
              ]}
            />

            <Bar dataKey="value" radius={[0, 8, 8, 0]}>
              {chartData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={COLORS[index % COLORS.length]}
                />
              ))}
            </Bar>

          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Clean custom legend */}
      <div className="bar-legend">
        {chartData.map((item, index) => (
          <div key={index} className="legend-item">
            <span
              className="legend-color"
              style={{ backgroundColor: COLORS[index % COLORS.length] }}
            />
            {item.name}
          </div>
        ))}
      </div>
    </div>
  )
}

export default Analysis
