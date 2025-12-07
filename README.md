# XIDS - Explainable Intrusion Detection System

XIDS is a full-stack security web application designed to detect network intrusions using Ensemble Machine Learning models. It goes beyond simple detection by providing **Explainable AI (XAI)** insights using SHAP (SHapley Additive exPlanations), helping security analysts understand *why* a specific traffic flow was flagged as an attack.

## Features

* **Ensemble Detection:** Combines Decision Trees, Random Forest, XGBoost, and LightGBM for high-accuracy threat detection.
* **Explainable AI (XAI):** Uses SHAP values to visualize feature contributions (e.g., why a packet was classified as DDoS vs Benign).
* **Interactive Threat Dashboard:** Visualizes attack distribution and status from uploaded data.
* **Simulation Mode:** Processes traffic row-by-row to mimic live monitoring.
* **PDF Reporting:** Auto-generates detailed attack summary reports.
* **AI Chatbot:** Built-in assistant to answer questions about security concepts and attack types.

## Tech Stack

**Backend:**
* **Python 3.10+**
* **FastAPI:** High-performance API framework.
* **ML Libraries:** Scikit-learn, XGBoost, LightGBM, SHAP.
* **Data Processing:** Pandas, NumPy.
* **Reporting:** ReportLab.

**Frontend:**
* **React (Vite):** Fast frontend build tool.
* **Visualization:** Recharts.
* **Styling:** CSS Modules.

---

## Installation & Setup

### 1. Prerequisites
Ensure you have the following installed:
* Python 3.8 or higher
* Node.js and npm

### 2. Backend Setup
Navigate to the backend directory and set up the Python environment.

```bash
cd backend

# Create a virtual environment (Recommended)
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install fastapi uvicorn pandas numpy scikit-learn xgboost lightgbm shap reportlab python-multipart joblib
```
Training the Models (Optional but Recommended): To generate the real .pkl model files:

```bash
# Ensure you have your dataset (CSVs) in a folder named 'data' inside backend/
# Then run the preprocessing and training scripts:
python preprocess.py
python "train scripts/ensemble_train.py"
```

### 3.Frontend Setup
Open a new terminal, navigate to the frontend directory, and install dependencies.

```bash
cd frontend
npm install
```

---

## Running the Application

### Step 1: Start the Backend
The frontend expects the backend to run on Port 5000

```bash
uvicorn app:app --host 0.0.0.0 --port 5000
```

###Step 2: Start the Frontend

```bash
# Inside the /frontend directory
npm run dev
```
The application will launch at http://localhost:5173.

## Usage Guide

* **Dashboard**: The landing page shows the system status.
* **Upload**: Drag and drop a network traffic CSV file (CIC-IDS format).
* **Analysis**: View the distribution of Benign vs. Malicious traffic.
* **Simulation**: Click "Start" to watch the system predict traffic row-by-row with explainability details.
* **Reports**: Click "Generate Full Attack Report (PDF)" to download a summary.
* **Chatbot**: Use the floating action button (bottom right) to ask about security terms (e.g., "What is DDoS?").

