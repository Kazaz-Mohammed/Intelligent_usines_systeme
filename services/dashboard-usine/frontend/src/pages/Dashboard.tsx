import React from 'react'
import { useDashboard } from '../context/DashboardContext'
import AssetCard from '../components/AssetCard'
import RULGauge from '../components/RULGauge'
import KPICard from '../components/KPICard'
import './Dashboard.css'

const Dashboard: React.FC = () => {
  const { assets, kpiSummary, loading } = useDashboard()

  if (loading) {
    return <div className="loading">Chargement...</div>
  }

  return (
    <div className="dashboard">
      <h1>Vue d'ensemble</h1>
      
      {/* KPIs */}
      <div className="kpi-grid">
        <KPICard
          title="MTBF"
          value={kpiSummary?.mtbf || 0}
          unit="h"
          color="#3498db"
        />
        <KPICard
          title="MTTR"
          value={kpiSummary?.mttr || 0}
          unit="h"
          color="#e74c3c"
        />
        <KPICard
          title="OEE"
          value={kpiSummary?.oee || 0}
          unit="%"
          color="#27ae60"
        />
        <KPICard
          title="Disponibilité"
          value={kpiSummary?.availability || 0}
          unit="%"
          color="#f39c12"
        />
      </div>

      {/* Assets Grid */}
      <div className="assets-section">
        <h2>Actifs surveillés ({assets.length})</h2>
        <div className="assets-grid">
          {assets.map((asset) => (
            <AssetCard key={asset.id} asset={asset} />
          ))}
        </div>
      </div>
    </div>
  )
}

export default Dashboard

