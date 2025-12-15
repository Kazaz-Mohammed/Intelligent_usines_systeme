import React from 'react'
import './KPICard.css'

interface KPICardProps {
  title: string
  value: number | string
  unit?: string
  trend?: number
  color?: string
}

const KPICard: React.FC<KPICardProps> = ({ title, value, unit, trend, color = '#3498db' }) => {
  return (
    <div className="kpi-card" style={{ borderTopColor: color }}>
      <h3 className="kpi-title">{title}</h3>
      <div className="kpi-value">
        {typeof value === 'number' ? value.toFixed(2) : value}
        {unit && <span className="kpi-unit">{unit}</span>}
      </div>
      {trend !== undefined && (
        <div className={`kpi-trend ${trend >= 0 ? 'positive' : 'negative'}`}>
          {trend >= 0 ? '↑' : '↓'} {Math.abs(trend).toFixed(1)}%
        </div>
      )}
    </div>
  )
}

export default KPICard

