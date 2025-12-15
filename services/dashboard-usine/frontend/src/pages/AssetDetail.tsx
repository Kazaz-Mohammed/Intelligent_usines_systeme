import React, { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { assetsApi, rulApi, anomaliesApi } from '../services/api'
import { AssetDetail as AssetDetailType } from '../types'
import RULGauge from '../components/RULGauge'
import './AssetDetail.css'

const AssetDetail: React.FC = () => {
  const { assetId } = useParams<{ assetId: string }>()
  const [asset, setAsset] = useState<AssetDetailType | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const loadAsset = async () => {
      if (!assetId) return
      try {
        const response = await assetsApi.getById(assetId)
        setAsset(response.data)
      } catch (error) {
        console.error('Error loading asset:', error)
      } finally {
        setLoading(false)
      }
    }
    loadAsset()
  }, [assetId])

  if (loading) {
    return <div className="loading">Chargement...</div>
  }

  if (!asset) {
    return <div className="error">Actif non trouvé</div>
  }

  return (
    <div className="asset-detail">
      <h1>{asset.name}</h1>
      
      <div className="asset-info">
        <div className="info-section">
          <h2>Informations générales</h2>
          <p><strong>ID:</strong> {asset.id}</p>
          <p><strong>Type:</strong> {asset.type}</p>
          <p><strong>Statut:</strong> {asset.status}</p>
          {asset.location && <p><strong>Localisation:</strong> {asset.location}</p>}
        </div>

        {asset.current_rul !== undefined && (
          <div className="rul-section">
            <h2>RUL (Remaining Useful Life)</h2>
            <RULGauge value={asset.current_rul} assetName={asset.name} />
          </div>
        )}

        <div className="info-section">
          <h2>Statistiques</h2>
          <p><strong>Anomalies détectées:</strong> {asset.anomaly_count}</p>
          {asset.last_anomaly && (
            <p><strong>Dernière anomalie:</strong> {new Date(asset.last_anomaly).toLocaleString()}</p>
          )}
        </div>
      </div>
    </div>
  )
}

export default AssetDetail

