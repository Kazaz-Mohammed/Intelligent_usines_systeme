import React from 'react'
import { Link } from 'react-router-dom'
import { Asset } from '../types'
import './AssetCard.css'

interface AssetCardProps {
  asset: Asset
}

const AssetCard: React.FC<AssetCardProps> = ({ asset }) => {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'operational':
        return '#27ae60'
      case 'warning':
        return '#f39c12'
      case 'critical':
        return '#e74c3c'
      case 'maintenance':
        return '#3498db'
      case 'offline':
        return '#95a5a6'
      default:
        return '#95a5a6'
    }
  }

  return (
    <Link to={`/assets/${asset.id}`} className="asset-card">
      <div className="asset-card-header">
        <h3>{asset.name}</h3>
        <span
          className="status-badge"
          style={{ backgroundColor: getStatusColor(asset.status) }}
        >
          {asset.status}
        </span>
      </div>
      <div className="asset-card-body">
        <p className="asset-id">ID: {asset.id}</p>
        <p className="asset-type">Type: {asset.type}</p>
        {asset.location && <p className="asset-location">📍 {asset.location}</p>}
      </div>
    </Link>
  )
}

export default AssetCard

