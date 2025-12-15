import React, { ReactNode } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { useDashboard } from '../context/DashboardContext'
import './DashboardLayout.css'

interface DashboardLayoutProps {
  children: ReactNode
}

const DashboardLayout: React.FC<DashboardLayoutProps> = ({ children }) => {
  const location = useLocation()
  const { isConnected } = useDashboard()

  return (
    <div className="dashboard-layout">
      <aside className="sidebar">
        <div className="sidebar-header">
          <h1>🏭 Dashboard Usine</h1>
          <div className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
            {isConnected ? '🟢 Connecté' : '🔴 Déconnecté'}
          </div>
        </div>
        <nav className="sidebar-nav">
          <Link
            to="/"
            className={location.pathname === '/' ? 'active' : ''}
          >
            📊 Vue d'ensemble
          </Link>
          <Link
            to="/anomalies"
            className={location.pathname === '/anomalies' ? 'active' : ''}
          >
            ⚠️ Anomalies
          </Link>
          <Link
            to="/maintenance"
            className={location.pathname === '/maintenance' ? 'active' : ''}
          >
            🔧 Maintenance
          </Link>
        </nav>
      </aside>
      <main className="main-content">
        {children}
      </main>
    </div>
  )
}

export default DashboardLayout

