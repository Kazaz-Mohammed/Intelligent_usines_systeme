import React, { useEffect, useState } from 'react'
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet'
import { AssetLocation } from '../../types'
import { gisApi } from '../../services/api'
import AssetMarker from './AssetMarker'
import 'leaflet/dist/leaflet.css'
import './FactoryMap.css'

const FactoryMap: React.FC = () => {
  const [locations, setLocations] = useState<AssetLocation[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const loadLocations = async () => {
      try {
        const response = await gisApi.getAssetLocations()
        setLocations(response.data.locations || [])
      } catch (error) {
        console.error('Error loading asset locations:', error)
      } finally {
        setLoading(false)
      }
    }
    loadLocations()
  }, [])

  if (loading) {
    return <div className="loading">Chargement de la carte...</div>
  }

  // Default center (can be configured)
  const center: [number, number] = [48.8566, 2.3522] // Paris coordinates as default

  return (
    <div className="factory-map">
      <MapContainer
        center={center}
        zoom={13}
        style={{ height: '600px', width: '100%' }}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {locations.map((location) => (
          <AssetMarker key={location.asset_id} location={location} />
        ))}
      </MapContainer>
    </div>
  )
}

export default FactoryMap

