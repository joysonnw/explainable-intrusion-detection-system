import React, { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import { FaPlay, FaPause } from 'react-icons/fa'
import './Simulation.css'

const API_URL = 'http://127.0.0.1:5000/api'

// only keep last 20 results visible
const MAX_RECORDS = 20

const Simulation = ({ data }) => {
  const [currentIndex, setCurrentIndex] = useState(0)
  const [isRunning, setIsRunning] = useState(false)
  const [results, setResults] = useState([])    // <-- CHANGED
  const [error, setError] = useState(null)
  const [dataReady, setDataReady] = useState(false)

  const intervalRef = useRef(null)
  const scrollRef = useRef(null)

  // Detect when data is ready
  useEffect(() => {
    if (data && data.length > 0) {
      setDataReady(true)
      setCurrentIndex(0)
      setResults([])   // clear old run
    } else {
      setDataReady(false)
    }
  }, [data])

  // Auto-scroll
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [results])

  // Main simulation loop
  useEffect(() => {
    if (isRunning && dataReady) {

      intervalRef.current = setInterval(async () => {

        if (currentIndex >= data.length) {
          setIsRunning(false)
          clearInterval(intervalRef.current)
          return
        }

        try {
          const response = await axios.post(
            `${API_URL}/predict_row`,
            data[currentIndex]
          )

          setResults(prev => {
            const newArr = [...prev, response.data]
            return newArr.slice(-MAX_RECORDS)   // keep last 20
          })

          setCurrentIndex(prev => prev + 1)
          setError(null)

        } catch (err) {
          console.error(err)
          setError("Prediction failed. Check backend.")
          setIsRunning(false)
        }

      }, 2500)

    }

    return () => clearInterval(intervalRef.current)

  }, [isRunning, currentIndex, data, dataReady])


  const handleStart = () => {
    if (!dataReady) return
    setIsRunning(true)
  }

  const handleStop = () => {
    setIsRunning(false)
    clearInterval(intervalRef.current)
  }

  const getBadgeClass = (label) => {
    if (!label) return ''
    if (label.toLowerCase().includes("benign")) return "benign"
    return "attack"
  }

  return (
    <div className="simulation-container">

      <div className="sim-header">
        <h3>Live Prediction Simulation</h3>

        {!isRunning ? (
          <button
            onClick={handleStart}
            className={`play-btn ${!dataReady ? 'disabled' : ''}`}
            disabled={!dataReady}
          >
            <FaPlay /> Start
          </button>
        ) : (
          <button onClick={handleStop} className="pause-btn">
            <FaPause /> Stop
          </button>
        )}
      </div>

      <div className="prediction-feed" ref={scrollRef}>

        {!dataReady && (
          <div className="sim-placeholder">
            Upload a CSV file to enable simulation...
          </div>
        )}

        {error && <div className="sim-error">{error}</div>}

        {dataReady && results.length === 0 && !error && (
          <div className="sim-placeholder">
            Ready. Click start to begin simulation.
          </div>
        )}

        {results.map((res, idx) => (
          <div className="sim-result-card" key={idx}>

            <div className={`prediction-badge ${getBadgeClass(res.prediction_label)}`}>
              {res.prediction_label}
            </div>

            <div className="prediction-details">
              <h4>Top Features</h4>

              <ul className="shap-list">
                {res.explanation?.map((item, index) => (
                  <li key={index}>
                    <span className="shap-feature">{item.feature}</span>
                    <span className="shap-value">{item.value}</span>
                    <span
                      className={`shap-contrib ${
                        item.contribution >= 0 ? 'positive' : 'negative'
                      }`}
                    >
                      {item.contribution.toFixed(6)}
                    </span>
                  </li>
                ))}
              </ul>
            </div>

            <div className="insight-box">
              <strong>Actionable Security Insight:</strong>
              <p>{res.insight}</p>
            </div>

          </div>
        ))}

      </div>

    </div>
  )
}

export default Simulation
