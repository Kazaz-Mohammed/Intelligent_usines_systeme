import React from 'react'
import { useDashboard } from '../context/DashboardContext'
import './MaintenancePlanning.css'

const MaintenancePlanning: React.FC = () => {
  const { interventions, loading } = useDashboard()

  if (loading) {
    return <div className="loading">Chargement...</div>
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return '#27ae60'
      case 'in_progress':
        return '#3498db'
      case 'planned':
        return '#f39c12'
      case 'overdue':
        return '#e74c3c'
      case 'cancelled':
        return '#95a5a6'
      default:
        return '#95a5a6'
    }
  }

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'urgent':
        return '#e74c3c'
      case 'high':
        return '#e67e22'
      case 'medium':
        return '#f39c12'
      case 'low':
        return '#3498db'
      default:
        return '#95a5a6'
    }
  }

  return (
    <div className="maintenance-planning">
      <h1>Planification de Maintenance</h1>
      <p className="intervention-count">Total: {interventions.length} interventions</p>
      
      <div className="interventions-list">
        {interventions.map((intervention) => (
          <div key={intervention.id || `${intervention.asset_id}-${intervention.title}`} className="intervention-card">
            <div className="intervention-header">
              <h3>{intervention.title}</h3>
              <div className="badges">
                <span
                  className="status-badge"
                  style={{ backgroundColor: getStatusColor(intervention.status) }}
                >
                  {intervention.status}
                </span>
                <span
                  className="priority-badge"
                  style={{ backgroundColor: getPriorityColor(intervention.priority) }}
                >
                  {intervention.priority}
                </span>
              </div>
            </div>
            <div className="intervention-body">
              <p><strong>Actif:</strong> {intervention.asset_id}</p>
              {intervention.description && <p>{intervention.description}</p>}
              {intervention.scheduled_start && (
                <p><strong>Début prévu:</strong> {new Date(intervention.scheduled_start).toLocaleString()}</p>
              )}
              {intervention.scheduled_end && (
                <p><strong>Fin prévue:</strong> {new Date(intervention.scheduled_end).toLocaleString()}</p>
              )}
              {intervention.assigned_to && (
                <p><strong>Assigné à:</strong> {intervention.assigned_to}</p>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default MaintenancePlanning

