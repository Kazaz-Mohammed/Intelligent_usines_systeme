"use client"

import { useParams } from "next/navigation"
import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { apiClient } from "@/lib/api"
import { useLatestRUL } from "@/hooks/useRUL"
import { useAnomalies } from "@/hooks/useAnomalies"
import { useInterventions } from "@/hooks/useInterventions"
import type { AssetDetail } from "@/types"
import { ArrowLeft } from "lucide-react"
import Link from "next/link"
import { Skeleton } from "@/components/ui/skeleton"

const statusColors = {
  operational: "status-operational",
  warning: "status-warning",
  critical: "status-critical",
  maintenance: "status-maintenance",
  offline: "status-offline",
}

export default function AssetDetailPage() {
  const params = useParams()
  const assetId = params.id as string
  const [asset, setAsset] = useState<AssetDetail | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  
  const { prediction: rul, isLoading: rulLoading } = useLatestRUL(assetId)
  const { anomalies, isLoading: anomaliesLoading } = useAnomalies({ asset_id: assetId, limit: 10 })
  const { interventions, isLoading: interventionsLoading } = useInterventions({ asset_id: assetId, limit: 10 })

  useEffect(() => {
    const fetchAsset = async () => {
      try {
        setIsLoading(true)
        const data = await apiClient.getAsset(assetId)
        setAsset(data)
        setError(null)
      } catch (err) {
        console.error("Failed to fetch asset:", err)
        setError("Failed to load asset details")
      } finally {
        setIsLoading(false)
      }
    }

    if (assetId) {
      fetchAsset()
    }
  }, [assetId])

  const getRULColor = (rul: number) => {
    if (rul > 365) return "text-green-600"
    if (rul > 180) return "text-yellow-600"
    return "text-red-600"
  }

  if (isLoading && !asset) {
    return (
      <div className="space-y-8">
        <Skeleton className="h-12 w-64" />
        <Skeleton className="h-64 w-full" />
      </div>
    )
  }

  if (error || !asset) {
    return (
      <div className="space-y-8">
        <Link href="/dashboard/assets">
          <Button variant="ghost" className="mb-4 gap-2">
            <ArrowLeft className="h-4 w-4" />
            Back to Assets
          </Button>
        </Link>
        <div className="bg-destructive/10 text-destructive p-4 rounded-lg">
          {error || "Asset not found"}
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <Link href="/dashboard/assets">
          <Button variant="ghost" className="mb-4 gap-2">
            <ArrowLeft className="h-4 w-4" />
            Back to Assets
          </Button>
        </Link>
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">{asset.name}</h1>
            <p className="text-muted-foreground mt-2">
              {asset.type} {asset.location ? `in ${asset.location}` : ""}
            </p>
          </div>
          <Badge className={`${statusColors[asset.status]} border-0 h-fit`}>{asset.status}</Badge>
        </div>
      </div>

      {/* Main grid */}
      <div className="grid gap-4 md:grid-cols-3">
        {/* RUL Gauge */}
        <Card className="border-border">
          <CardHeader>
            <CardTitle className="text-base">Remaining Useful Life</CardTitle>
          </CardHeader>
          <CardContent>
            {rulLoading ? (
              <Skeleton className="h-32 w-full" />
            ) : rul ? (
              <div className="text-center">
                <p className={`text-4xl font-bold ${getRULColor(rul.rul_prediction)}`}>
                  {rul.rul_prediction.toFixed(0)} h
                </p>
                <p className="text-xs text-muted-foreground mt-2">
                  Confidence: {((rul.confidence_level || 0) * 100).toFixed(0)}%
                </p>
                <div className="mt-4 space-y-1 text-xs">
                  <div className="flex justify-between">
                    <span>CI Lower:</span>
                    <span>{rul.confidence_interval_lower?.toFixed(0) || "-"}h</span>
                  </div>
                  <div className="flex justify-between">
                    <span>CI Upper:</span>
                    <span>{rul.confidence_interval_upper?.toFixed(0) || "-"}h</span>
                  </div>
                </div>
              </div>
            ) : (
              <p className="text-center text-muted-foreground">No RUL data available</p>
            )}
          </CardContent>
        </Card>

        {/* Anomaly Summary */}
        <Card className="border-border">
          <CardHeader>
            <CardTitle className="text-base">Anomalies Detected</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-4xl font-bold">{asset.anomaly_count || 0}</p>
            <p className="text-xs text-muted-foreground mt-2">
              Last: {asset.last_anomaly ? new Date(asset.last_anomaly).toLocaleDateString() : "Never"}
            </p>
          </CardContent>
        </Card>

        {/* Maintenance Info */}
        <Card className="border-border">
          <CardHeader>
            <CardTitle className="text-base">Maintenance Schedule</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div>
              <p className="text-xs text-muted-foreground">Last Maintenance</p>
              <p className="font-medium">
                {asset.last_maintenance ? new Date(asset.last_maintenance).toLocaleDateString() : "Never"}
              </p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Next Scheduled</p>
              <p className="font-medium">
                {asset.next_maintenance ? new Date(asset.next_maintenance).toLocaleDateString() : "Not scheduled"}
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Anomalies Table */}
      <Card className="border-border">
        <CardHeader>
          <CardTitle>Recent Anomalies</CardTitle>
        </CardHeader>
        <CardContent>
          {anomaliesLoading ? (
            <div className="space-y-3">
              {[1, 2, 3].map((i) => (
                <Skeleton key={i} className="h-16 w-full" />
              ))}
            </div>
          ) : anomalies.length === 0 ? (
            <p className="text-sm text-muted-foreground text-center py-4">No anomalies detected</p>
          ) : (
            <div className="space-y-3">
              {anomalies.map((anomaly, idx) => (
                <div key={anomaly.id || idx} className="flex items-center justify-between p-3 border border-border rounded-lg">
                  <div>
                    <p className="font-medium">{new Date(anomaly.timestamp).toLocaleString()}</p>
                    <p className="text-xs text-muted-foreground">Score: {(anomaly.anomaly_score || 0).toFixed(2)}</p>
                  </div>
                  <Badge className={`severity-${anomaly.severity} border-0`}>{anomaly.severity}</Badge>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Active Interventions */}
      <Card className="border-border">
        <CardHeader>
          <CardTitle>Active Interventions</CardTitle>
        </CardHeader>
        <CardContent>
          {interventionsLoading ? (
            <div className="space-y-3">
              {[1, 2, 3].map((i) => (
                <Skeleton key={i} className="h-16 w-full" />
              ))}
            </div>
          ) : interventions.length === 0 ? (
            <p className="text-sm text-muted-foreground text-center py-4">No active interventions</p>
          ) : (
            <div className="space-y-3">
              {interventions.map((intervention, idx) => (
                <div key={intervention.id || idx} className="flex items-center justify-between p-3 border border-border rounded-lg">
                  <div>
                    <p className="font-medium">{intervention.title}</p>
                    <p className="text-xs text-muted-foreground">{intervention.description || "No description"}</p>
                  </div>
                  <Badge variant="outline">{intervention.status}</Badge>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
