import React from 'react'
import { Marker, Popup } from 'react-leaflet'
import { Icon } from 'leaflet'
import { AssetLocation } from '../../types'
import { Link } from 'react-router-dom'

interface AssetMarkerProps {
  location: AssetLocation
}

const AssetMarker: React.FC<AssetMarkerProps> = ({ location }) => {
  // Create custom icon
  const assetIcon = new Icon({
    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowSize: [41, 41],
  })

  return (
    <Marker
      position={[location.point.lat, location.point.lon]}
      icon={assetIcon}
    >
      <Popup>
        <div>
          <h3>{location.asset_name}</h3>
          <p><strong>ID:</strong> {location.asset_id}</p>
          {location.building && <p><strong>Bâtiment:</strong> {location.building}</p>}
          {location.zone && <p><strong>Zone:</strong> {location.zone}</p>}
          <Link to={`/assets/${location.asset_id}`}>Voir détails</Link>
        </div>
      </Popup>
    </Marker>
  )
}

export default AssetMarker

