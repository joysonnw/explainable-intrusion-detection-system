import React, { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import axios from 'axios'
import {
  FaUpload,
  FaSpinner,
  FaCheckCircle,
  FaExclamationCircle,
  FaFilePdf
} from 'react-icons/fa'
import './Upload.css'

const API_URL = 'http://127.0.0.1:5000/api'

const Upload = ({ onUploadSuccess }) => {
  const [status, setStatus] = useState('idle')
  const [message, setMessage] = useState('Drag & drop a .csv file here, or click to select')
  const [uploadedFile, setUploadedFile] = useState(null)

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0]
    if (!file) {
      setStatus('error')
      setMessage('File type not accepted. Please upload a .csv file.')
      return
    }

    setUploadedFile(file)

    const formData = new FormData()
    formData.append('file', file)

    setStatus('uploading')
    setMessage(`Uploading ${file.name}...`)

    axios.post(`${API_URL}/upload`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
      .then(response => {
        setStatus('success')
        setMessage(`Successfully processed ${response.data.fileName} (${response.data.rowCount} rows)`)
        onUploadSuccess(response.data)
      })
      .catch(error => {
        setStatus('error')
        const errorMsg = error.response?.data?.error || 'An unknown error occurred.'
        setMessage(`Upload failed: ${errorMsg}`)
      })

  }, [onUploadSuccess])

  const generatePDF = async () => {
    if (!uploadedFile) return

    const formData = new FormData()
    formData.append("file", uploadedFile)

    try {
      setStatus('uploading')
      setMessage("Generating Full Attack Report (PDF)...")

      const res = await axios.post(
        `${API_URL}/generate-full-report`,
        formData,
        { responseType: "blob" }
      )

      const url = window.URL.createObjectURL(new Blob([res.data]))
      const link = document.createElement("a")
      link.href = url
      link.setAttribute("download", "Full_Attack_Report.pdf")
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)

      setStatus('success')
      setMessage("PDF Report generated and downloaded.")

    } catch (err) {
      console.error(err)
      setStatus('error')
      setMessage("Failed to generate PDF report.")
    }
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'text/csv': ['.csv'] },
    multiple: false
  })

  const getIcon = () => {
    switch (status) {
      case 'uploading':
        return <FaSpinner className="spin" />
      case 'success':
        return <FaCheckCircle style={{ color: 'var(--color-success)' }} />
      case 'error':
        return <FaExclamationCircle style={{ color: 'var(--color-danger)' }} />
      default:
        return <FaUpload />
    }
  }

  return (
    <>
      <div {...getRootProps()} className={`upload-box ${isDragActive ? 'active' : ''} ${status}`}>
        <input {...getInputProps()} />
        <div className="upload-icon">{getIcon()}</div>
        <p>{message}</p>
      </div>

      {uploadedFile && status === "success" && (
        <button
          className="btn-danger"
          onClick={generatePDF}
          style={{ marginTop: '12px', width: '100%' }}
        >
          <FaFilePdf /> Generate Full Attack Report (PDF)
        </button>
      )}
    </>
  )
}

export default Upload
