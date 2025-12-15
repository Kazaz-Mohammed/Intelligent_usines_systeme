import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { DashboardProvider } from './context/DashboardContext'
import DashboardLayout from './components/DashboardLayout'
import Dashboard from './pages/Dashboard'
import AssetDetail from './pages/AssetDetail'
import AnomalyAnalysis from './pages/AnomalyAnalysis'
import MaintenancePlanning from './pages/MaintenancePlanning'
import './App.css'

function App() {
  return (
    <DashboardProvider>
      <Router>
        <DashboardLayout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/assets/:assetId" element={<AssetDetail />} />
            <Route path="/anomalies" element={<AnomalyAnalysis />} />
            <Route path="/maintenance" element={<MaintenancePlanning />} />
          </Routes>
        </DashboardLayout>
      </Router>
    </DashboardProvider>
  )
}

export default App

