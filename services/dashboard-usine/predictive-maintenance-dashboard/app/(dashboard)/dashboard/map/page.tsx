"use client"

import dynamic from "next/dynamic"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useAssets } from "@/hooks/useAssets"
import { Skeleton } from "@/components/ui/skeleton"
import { Badge } from "@/components/ui/badge"
import "leaflet/dist/leaflet.css"

// Dynamically import the map component (client-side only)
const AssetMap = dynamic(() => import("@/components/charts/asset-map").then((mod) => ({ default: mod.AssetMap })), {
  ssr: false,
  loading: () => <Skeleton className="h-96 w-full" />
})

export default function MapPage() {
  const { assets, isLoading, error } = useAssets({ page_size: 100 })

  // Filter assets that have coordinates
  const assetsWithCoordinates = assets.filter(
    (asset) => asset.coordinates && asset.coordinates.lat && asset.coordinates.lon
  )
  
  // All assets (for the list view)
  const allAssets = assets

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Asset Map</h1>
        <p className="text-muted-foreground mt-2">View your assets on a geographic map</p>
      </div>

      {error && (
        <div className="bg-destructive/10 text-destructive p-4 rounded-lg">
          {error}
        </div>
      )}

      <Card className="border-border">
        <CardHeader>
          <CardTitle>Asset Locations</CardTitle>
          {!isLoading && assetsWithCoordinates.length > 0 && (
            <p className="text-xs text-muted-foreground mt-1">
              {assetsWithCoordinates.length} asset{assetsWithCoordinates.length !== 1 ? 's' : ''} displayed on map
            </p>
          )}
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <Skeleton className="h-96 w-full" />
          ) : assetsWithCoordinates.length === 0 ? (
            <div className="bg-gradient-to-br from-slate-100 to-slate-200 dark:from-slate-800 dark:to-slate-900 rounded-lg h-96 flex items-center justify-center border border-dashed border-border">
              <div className="text-center">
                <p className="text-muted-foreground font-medium">No Assets with Coordinates</p>
                <p className="text-xs text-muted-foreground mt-1">
                  Add coordinates to your assets to view them on the map
                </p>
              </div>
            </div>
          ) : (
            <AssetMap assets={assetsWithCoordinates} height="500px" />
          )}
        </CardContent>
      </Card>

      <Card className="border-border">
        <CardHeader>
          <CardTitle>Asset Locations List</CardTitle>
          <p className="text-xs text-muted-foreground mt-1">
            {assetsWithCoordinates.length} of {allAssets.length} assets have coordinates
          </p>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-3">
              {[1, 2, 3].map((i) => (
                <Skeleton key={i} className="h-16 w-full" />
              ))}
            </div>
          ) : allAssets.length === 0 ? (
            <div className="text-center py-8">
              <p className="text-muted-foreground">No assets found</p>
            </div>
          ) : (
            <div className="space-y-3">
              {allAssets.map((asset) => {
                const hasCoordinates = asset.coordinates && asset.coordinates.lat && asset.coordinates.lon
                return (
                  <div
                    key={asset.id}
                    className="flex items-center justify-between p-3 border border-border rounded-lg hover:bg-muted/50"
                  >
                    <div className="flex-1">
                      <div className="flex items-center gap-2 flex-wrap">
                        <p className="font-medium">{asset.name || asset.id}</p>
                        <Badge variant="outline" className="text-xs">
                          {asset.type}
                        </Badge>
                        <Badge
                          className={`text-xs ${
                            asset.status === "operational"
                              ? "bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-100"
                              : asset.status === "warning"
                              ? "bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-100"
                              : asset.status === "critical"
                              ? "bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-100"
                              : "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-100"
                          }`}
                        >
                          {asset.status}
                        </Badge>
                        {!hasCoordinates && (
                          <Badge variant="outline" className="text-xs text-muted-foreground">
                            No coordinates
                          </Badge>
                        )}
                      </div>
                      {asset.location && (
                        <p className="text-xs text-muted-foreground mt-1">{asset.location}</p>
                      )}
                    </div>
                    {hasCoordinates ? (
                      <div className="text-xs font-mono text-muted-foreground ml-4">
                        {asset.coordinates.lat.toFixed(4)}, {asset.coordinates.lon.toFixed(4)}
                      </div>
                    ) : (
                      <div className="text-xs text-muted-foreground ml-4 italic">
                        Coordinates not set
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
