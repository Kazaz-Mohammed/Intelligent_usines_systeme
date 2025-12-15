import React from 'react'
import { useDashboard } from '../context/DashboardContext'
import './AnomalyAnalysis.css'

const AnomalyAnalysis: React.FC = () => {
  const { anomalies, loading } = useDashboard()

  if (loading) {
    return <div className="loading">Chargement...</div>
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return '#e74c3c'
      case 'high':
        return '#e67e22'
      case 'medium':
        return '#f39c12'
      case 'low':
        return '#f1c40f'
      default:
        return '#95a5a6'
    }
  }

  return (
    <div className="anomaly-analysis">
      <h1>Analyse des Anomalies</h1>
      <p className="anomaly-count">Total: {anomalies.length} anomalies détectées</p>
      
      <div className="anomalies-list">
        {anomalies.map((anomaly) => (
          <div key={anomaly.id || `${anomaly.asset_id}-${anomaly.timestamp}`} className="anomaly-card">
            <div className="anomaly-header">
              <h3>Actif: {anomaly.asset_id}</h3>
              <span
                className="severity-badge"
                style={{ backgroundColor: getSeverityColor(anomaly.severity) }}
              >
                {anomaly.severity}
              </span>
            </div>
            <div className="anomaly-body">
              <p><strong>Timestamp:</strong> {new Date(anomaly.timestamp).toLocaleString()}</p>
              {anomaly.anomaly_score !== undefined && (
                <p><strong>Score:</strong> {anomaly.anomaly_score.toFixed(3)}</p>
              )}
              {anomaly.model_used && (
                <p><strong>Modèle:</strong> {anomaly.model_used}</p>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default AnomalyAnalysis

