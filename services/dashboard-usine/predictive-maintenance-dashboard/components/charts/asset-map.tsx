"use client"

import { useEffect, useRef } from "react"
import type { Asset } from "@/types"
import "leaflet/dist/leaflet.css"

interface AssetMapProps {
  assets: Asset[]
  height?: string
}

export function AssetMap({ assets, height = "400px" }: AssetMapProps) {
  const mapRef = useRef<HTMLDivElement>(null)
  const mapInstanceRef = useRef<any>(null)

  useEffect(() => {
    if (!mapRef.current || assets.length === 0) return

    // Dynamically import Leaflet only on client side
    const initMap = async () => {
      const L = await import("leaflet")

      // Clean up existing map
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove()
      }

      // Calculate center from assets
      const validAssets = assets.filter(
        (asset) => asset.coordinates?.lat && asset.coordinates?.lon
      )

      if (validAssets.length === 0) return

      const avgLat =
        validAssets.reduce((sum, a) => sum + (a.coordinates?.lat || 0), 0) /
        validAssets.length
      const avgLon =
        validAssets.reduce((sum, a) => sum + (a.coordinates?.lon || 0), 0) /
        validAssets.length

      // Create map
      const map = L.map(mapRef.current).setView([avgLat, avgLon], 13)

      // Add OpenStreetMap tiles
      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution:
          '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: 19,
      }).addTo(map)

      // Add markers for each asset
      validAssets.forEach((asset) => {
        if (!asset.coordinates?.lat || !asset.coordinates?.lon) return

        // Choose marker color based on status
        const getMarkerColor = (status: string) => {
          switch (status) {
            case "operational":
              return "green"
            case "warning":
              return "orange"
            case "critical":
              return "red"
            case "maintenance":
              return "blue"
            default:
              return "gray"
          }
        }

        // Create custom icon
        const icon = L.divIcon({
          className: "custom-marker",
          html: `<div style="
            background-color: ${getMarkerColor(asset.status)};
            width: 20px;
            height: 20px;
            border-radius: 50%;
            border: 2px solid white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
          "></div>`,
          iconSize: [20, 20],
          iconAnchor: [10, 10],
        })

        const marker = L.marker(
          [asset.coordinates.lat, asset.coordinates.lon],
          { icon }
        ).addTo(map)

        // Add popup with asset info
        marker.bindPopup(`
          <div style="min-width: 200px;">
            <h3 style="margin: 0 0 8px 0; font-weight: bold;">${asset.name || asset.id}</h3>
            <p style="margin: 4px 0;"><strong>Type:</strong> ${asset.type}</p>
            <p style="margin: 4px 0;"><strong>Status:</strong> ${asset.status}</p>
            ${asset.location ? `<p style="margin: 4px 0;"><strong>Location:</strong> ${asset.location}</p>` : ""}
            <p style="margin: 4px 0; font-size: 0.85em; color: #666;">
              ${asset.coordinates.lat.toFixed(4)}, ${asset.coordinates.lon.toFixed(4)}
            </p>
          </div>
        `)
      })

      // Fit map to show all markers
      if (validAssets.length > 1) {
        const bounds = L.latLngBounds(
          validAssets.map((a) => [
            a.coordinates!.lat,
            a.coordinates!.lon,
          ])
        )
        map.fitBounds(bounds, { padding: [50, 50] })
      }

      mapInstanceRef.current = map
    }

    initMap()

    // Cleanup
    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove()
        mapInstanceRef.current = null
      }
    }
  }, [assets])

  if (assets.length === 0) {
    return (
      <div
        className="flex items-center justify-center border border-dashed border-border rounded-lg"
        style={{ height }}
      >
        <p className="text-muted-foreground">No assets with coordinates</p>
      </div>
    )
  }

  return (
    <div
      ref={mapRef}
      className="w-full rounded-lg border border-border"
      style={{ height, zIndex: 0 }}
    />
  )
}

