"use client"

import { useState, useMemo } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { useRUL } from "@/hooks/useRUL"
import { RULTrendChart } from "@/components/charts/rul-trend-chart"
import { Skeleton } from "@/components/ui/skeleton"

export default function RULPage() {
  const [selectedAsset, setSelectedAsset] = useState<string>("all")
  const { predictions, isLoading, error } = useRUL({
    asset_id: selectedAsset !== "all" ? selectedAsset : undefined,
    limit: 100,
  })

  const filteredPredictions = useMemo(() => {
    if (!predictions || !Array.isArray(predictions)) return []
    if (selectedAsset === "all") return predictions
    return predictions.filter((p) => p.asset_id === selectedAsset)
  }, [predictions, selectedAsset])

  // Get unique asset IDs for filter dropdown
  const assetIds = useMemo(() => {
    if (!predictions || !Array.isArray(predictions)) return []
    const unique = Array.from(new Set(predictions.map((p) => p.asset_id)))
    return unique
  }, [predictions])

  const getRULStatus = (rul: number) => {
    if (rul > 365) return { color: "bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-100", label: "Good" }
    if (rul > 180)
      return { color: "bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-100", label: "Fair" }
    return { color: "bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-100", label: "Critical" }
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight">RUL Predictions</h1>
        <p className="text-muted-foreground mt-2">Remaining Useful Life predictions for your assets</p>
      </div>

      {error && (
        <div className="bg-destructive/10 text-destructive p-4 rounded-lg">
          {error}
        </div>
      )}

      {/* Trend Chart */}
      {selectedAsset !== "all" && filteredPredictions && filteredPredictions.length > 0 && (
        <RULTrendChart
          data={filteredPredictions.map((p) => ({
            date: new Date(p.timestamp).toLocaleDateString(),
            rul: p.rul_prediction,
            lower: p.confidence_interval_lower || p.rul_prediction * 0.9,
            upper: p.confidence_interval_upper || p.rul_prediction * 1.1,
          }))}
          title={`RUL Trend - Asset ${selectedAsset}`}
        />
      )}

      {/* Filter and Table */}
      <Card className="border-border">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>All Predictions</CardTitle>
            <Select value={selectedAsset} onValueChange={setSelectedAsset}>
              <SelectTrigger className="w-40">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Assets</SelectItem>
                {assetIds.map((id) => (
                  <SelectItem key={id} value={id}>
                    Asset {id}
                  </SelectItem>
                ))}
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
                  <TableHead>RUL Prediction (h)</TableHead>
                  <TableHead>Confidence</TableHead>
                  <TableHead>CI Lower</TableHead>
                  <TableHead>CI Upper</TableHead>
                  <TableHead>Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {isLoading ? (
                  <TableRow>
                    <TableCell colSpan={6} className="text-center py-8">
                      <Skeleton className="h-4 w-full" />
                    </TableCell>
                  </TableRow>
                ) : !filteredPredictions || filteredPredictions.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={6} className="text-center py-8">
                      <p className="text-muted-foreground">No RUL predictions found</p>
                    </TableCell>
                  </TableRow>
                ) : (
                  filteredPredictions.map((pred, idx) => {
                    const status = getRULStatus(pred.rul_prediction)
                    return (
                      <TableRow key={pred.id || idx} className="border-border hover:bg-muted/50">
                        <TableCell className="font-medium">{pred.asset_id}</TableCell>
                        <TableCell className="font-bold">{pred.rul_prediction.toFixed(0)}</TableCell>
                        <TableCell>{((pred.confidence_level || 0) * 100).toFixed(0)}%</TableCell>
                        <TableCell>{pred.confidence_interval_lower?.toFixed(0) || "-"}</TableCell>
                        <TableCell>{pred.confidence_interval_upper?.toFixed(0) || "-"}</TableCell>
                        <TableCell>
                          <Badge className={`${status.color} border-0`}>{status.label}</Badge>
                        </TableCell>
                      </TableRow>
                    )
                  })
                )}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
