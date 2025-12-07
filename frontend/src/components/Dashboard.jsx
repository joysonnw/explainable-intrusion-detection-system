import React, { useState } from 'react'
import Upload from './Upload'
import SummaryCard from './SummaryCard'
import Simulation from './Simulation'
import Analysis from './Analysis'
import { FaFileUpload, FaCogs, FaChartPie, FaExclamationTriangle } from 'react-icons/fa'
import './Dashboard.css'

const Dashboard = () => {
  const [analysisData, setAnalysisData] = useState([])
  const [simulationData, setSimulationData] = useState(null)
  const [summary, setSummary] = useState({
    fileName: 'N/A',
    rowCount: 0,
    attacks: 0,
  })

  const onUploadSuccess = (data) => {
    setAnalysisData(data.analysis)

    setSimulationData(data.previewData)

    setSummary({
      fileName: data.fileName,
      rowCount: data.rowCount,
      attacks: data.analysis
        .filter(a => a.name !== 'BENIGN')
        .reduce((sum, a) => sum + a.value, 0),
    })
  }

  return (
    <div className="dashboard-grid">

      <div className="grid-item span-2">
        <Upload onUploadSuccess={onUploadSuccess} />
      </div>

      <div className="grid-item">
        <SummaryCard
          icon={<FaExclamationTriangle size={24} />}
          title="Total Attacks"
          value={summary.attacks}
          color="var(--color-danger)"
        />
      </div>

      <div className="grid-item">
        <SummaryCard
          icon={<FaFileUpload size={24} />}
          title="File Processed"
          value={summary.fileName}
          color="var(--color-accent)"
        />
      </div>

      <div className="grid-item span-2">
        <Analysis data={analysisData} />
      </div>

      <div className="grid-item span-2">
        <Simulation data={simulationData} />
      </div>

    </div>
  )
}

export default Dashboard
