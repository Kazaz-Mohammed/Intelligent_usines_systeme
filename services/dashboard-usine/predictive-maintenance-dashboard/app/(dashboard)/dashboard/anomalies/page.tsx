"use client"

import { useState, useMemo } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { useAnomalies } from "@/hooks/useAnomalies"
import { AnomalyHeatmap } from "@/components/charts/anomaly-heatmap"
import { Skeleton } from "@/components/ui/skeleton"

const severityColors = {
  low: "severity-low",
  medium: "severity-medium",
  high: "severity-high",
  critical: "severity-critical",
}

export default function AnomaliesPage() {
  const [severityFilter, setSeverityFilter] = useState<string>("all")
  const { anomalies, isLoading, error } = useAnomalies({
    limit: 100,
    criticality: severityFilter !== "all" ? severityFilter : undefined,
  })

  const filteredAnomalies = useMemo(() => {
    if (severityFilter === "all") return anomalies
    return anomalies.filter((a) => a.severity === severityFilter)
  }, [anomalies, severityFilter])

  const criticalCount = useMemo(() => anomalies.filter((a) => a.severity === "critical").length, [anomalies])
  const highCount = useMemo(() => anomalies.filter((a) => a.severity === "high").length, [anomalies])

  // Compute heatmap data at the top level (hooks must not be inside JSX)
  const heatmapData = useMemo(() => {
    const assetSeverityMap: Record<string, Record<string, number>> = {}
    anomalies.forEach((anomaly) => {
      if (!assetSeverityMap[anomaly.asset_id]) {
        assetSeverityMap[anomaly.asset_id] = { low: 0, medium: 0, high: 0, critical: 0 }
      }
      const severity = anomaly.severity as keyof typeof assetSeverityMap[string]
      assetSeverityMap[anomaly.asset_id][severity] =
        (assetSeverityMap[anomaly.asset_id][severity] || 0) + 1
    })
    return Object.entries(assetSeverityMap).map(([asset, counts]) => ({
      asset,
      low: counts.low || 0,
      medium: counts.medium || 0,
      high: counts.high || 0,
      critical: counts.critical || 0,
    }))
  }, [anomalies])

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Anomalies</h1>
        <p className="text-muted-foreground mt-2">Monitor detected anomalies across your assets</p>
      </div>

      {error && (
        <div className="bg-destructive/10 text-destructive p-4 rounded-lg">
          {error}
        </div>
      )}

      {/* Summary Cards */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card className="border-border">
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Total Anomalies</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? <Skeleton className="h-8 w-16" /> : <p className="text-3xl font-bold">{anomalies.length}</p>}
          </CardContent>
        </Card>
        <Card className="border-border">
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Critical</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? <Skeleton className="h-8 w-16" /> : <p className="text-3xl font-bold text-red-600">{criticalCount}</p>}
          </CardContent>
        </Card>
        <Card className="border-border">
          <CardHeader className="pb-3">
            <CardTitle className="text-base">High Severity</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? <Skeleton className="h-8 w-16" /> : <p className="text-3xl font-bold text-orange-600">{highCount}</p>}
          </CardContent>
        </Card>
      </div>

      {/* Anomaly Heatmap */}
      {!isLoading && anomalies.length > 0 && (
        <Card className="border-border">
          <CardHeader>
            <CardTitle>Anomaly Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <AnomalyHeatmap data={heatmapData} />
          </CardContent>
        </Card>
      )}

      {/* Filter and Table */}
      <Card className="border-border">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Anomaly List</CardTitle>
            <Select value={severityFilter} onValueChange={setSeverityFilter}>
              <SelectTrigger className="w-40">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Severities</SelectItem>
                <SelectItem value="low">Low</SelectItem>
                <SelectItem value="medium">Medium</SelectItem>
                <SelectItem value="high">High</SelectItem>
                <SelectItem value="critical">Critical</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardHeader>
        <CardContent>
          <div className="border border-border rounded-lg overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow className="border-border hover:bg-transparent">
                  <TableHead>Asset ID</TableHead>
                  <TableHead>Timestamp</TableHead>
                  <TableHead>Severity</TableHead>
                  <TableHead>Anomaly Score</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {isLoading ? (
                  <TableRow>
                    <TableCell colSpan={4} className="text-center py-8">
                      <Skeleton className="h-4 w-full" />
                    </TableCell>
                  </TableRow>
                ) : filteredAnomalies.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={4} className="text-center py-8">
                      <p className="text-muted-foreground">No anomalies found</p>
                    </TableCell>
                  </TableRow>
                ) : (
                  filteredAnomalies.map((anomaly, idx) => (
                    <TableRow key={anomaly.id || idx} className="border-border hover:bg-muted/50">
                      <TableCell className="font-medium">{anomaly.asset_id}</TableCell>
                      <TableCell className="text-sm">{new Date(anomaly.timestamp).toLocaleString()}</TableCell>
                      <TableCell>
                        <Badge className={`${severityColors[anomaly.severity]} border-0`}>{anomaly.severity}</Badge>
                      </TableCell>
                      <TableCell className="font-medium">{(anomaly.anomaly_score || 0).toFixed(2)}</TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
