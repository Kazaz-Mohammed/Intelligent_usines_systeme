"use client"

import { useEffect, useState, useMemo } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ArrowUp, ArrowDown, TrendingUp } from "lucide-react"
import { useWebSocket } from "@/hooks/useWebSocket"
import { StatusPieChart } from "@/components/charts/status-pie-chart"
import { KPIGauge } from "@/components/charts/kpi-gauge"
import { useKPISummary } from "@/hooks/useKPIs"
import { useAssets } from "@/hooks/useAssets"
import { useAnomalies } from "@/hooks/useAnomalies"
import { useInterventions } from "@/hooks/useInterventions"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"
import { Skeleton } from "@/components/ui/skeleton"

export default function DashboardPage() {
  const { kpis, isLoading: kpisLoading, refetch: refetchKPIs } = useKPISummary({ days: 30 })
  const { assets, isLoading: assetsLoading, refetch: refetchAssets } = useAssets({ page_size: 100 })
  
  // Memoize the anomaly params for trend chart
  // Don't use start_date filter since we group by created_at (detection time)
  // but the API filters by timestamp (sensor data time) - they don't match
  const anomalyParams = useMemo(() => {
    return {
      limit: 5000, // Get all anomalies for accurate trend chart
      is_anomaly: true // Only fetch actual anomalies
    }
  }, []) // Empty dependency array - calculate once on mount
  
  // Separate params for recent anomalies - fetch most recent without date filter
  // Backend sorts by timestamp DESC, so we get newest first
  // Removed is_anomaly filter to ensure we get all recent detections
  const recentAnomalyParams = useMemo(() => {
    return {
      limit: 100, // Get more to ensure we have recent ones
      // No date filter, no is_anomaly filter - just get the most recent anomalies
      // We'll filter client-side to only show is_anomaly=true
    }
  }, [])
  
  const { anomalies, isLoading: anomaliesLoading, refetch: refetchAnomalies } = useAnomalies(anomalyParams)
  const { anomalies: recentAnomalies, isLoading: recentAnomaliesLoading, refetch: refetchRecentAnomalies } = useAnomalies(recentAnomalyParams)
  const { interventions, isLoading: interventionsLoading, refetch: refetchInterventions } = useInterventions({ limit: 10 })
  
  // Exclude recentAnomaliesLoading from main isLoading to prevent flickering
  const isLoading = kpisLoading || assetsLoading || anomaliesLoading || interventionsLoading

  const { isConnected } = useWebSocket({
    onMessage: (message) => {
      console.log("Real-time update received:", message)
      // Refresh data on real-time updates
      if (message.type === "feature_update" || message.type === "anomaly_detected" || message.type === "rul_prediction") {
        refetchKPIs()
        refetchAssets()
        refetchAnomalies()
        refetchRecentAnomalies() // Also refresh recent anomalies
        refetchInterventions()
      }
    },
  })

  // Auto-refresh recent anomalies every 30 seconds to show new detections
  // Use silent refresh - don't show loading state
  useEffect(() => {
    // Initial fetch delay to ensure data is loaded after mount
    const timeout = setTimeout(() => {
      refetchRecentAnomalies()
    }, 2000) // Refresh after 2 seconds on mount
    
    const interval = setInterval(() => {
      // Silently refresh without showing loading state
      refetchRecentAnomalies()
    }, 30000) // Refresh every 30 seconds

    return () => {
      clearTimeout(timeout)
      clearInterval(interval)
    }
  }, [refetchRecentAnomalies])

  // Calculate status distribution from assets
  const statusData = useMemo(() => {
    if (!assets || !Array.isArray(assets)) {
      return []
    }
    const statusCounts = assets.reduce((acc, asset) => {
      acc[asset.status] = (acc[asset.status] || 0) + 1
      return acc
    }, {} as Record<string, number>)

    const colors: Record<string, string> = {
      operational: "var(--color-chart-2)",
      warning: "var(--color-chart-4)",
      critical: "var(--color-chart-5)",
      maintenance: "var(--color-chart-3)",
      offline: "var(--color-chart-1)",
    }

    return Object.entries(statusCounts).map(([name, value]) => ({
      name: name.charAt(0).toUpperCase() + name.slice(1),
      value,
      color: colors[name] || "var(--color-chart-1)",
    }))
  }, [assets])

  // Generate anomaly trend from actual anomalies data (last 7 days by actual date)
  const anomalyTrendData = useMemo(() => {
    const counts: Record<string, number> = {}
    
    // Initialize last 7 days with 0 (including today)
    const today = new Date()
    // Ensure we include today by going from 6 days ago to today (7 days total)
    for (let i = 6; i >= 0; i--) {
      const date = new Date(today)
      date.setDate(date.getDate() - i)
      // Use UTC to avoid timezone issues
      const year = date.getUTCFullYear()
      const month = String(date.getUTCMonth() + 1).padStart(2, '0')
      const day = String(date.getUTCDate()).padStart(2, '0')
      const dateKey = `${year}-${month}-${day}` // YYYY-MM-DD format
      counts[dateKey] = 0
    }
    
    // Count anomalies per detection date (use created_at when available, fall back to timestamp)
    if (anomalies && Array.isArray(anomalies)) {
      anomalies.forEach(anomaly => {
        try {
          // Use created_at (when detected) for trend, not timestamp (sensor data time)
          const detectionTime = anomaly.created_at || anomaly.timestamp
          const date = new Date(detectionTime)
          // Convert to UTC date string to match the initialized dates
          const year = date.getUTCFullYear()
          const month = String(date.getUTCMonth() + 1).padStart(2, '0')
          const day = String(date.getUTCDate()).padStart(2, '0')
          const dateKey = `${year}-${month}-${day}` // YYYY-MM-DD format
          
          // Always add to counts, even if not in initialized range (for today's anomalies)
          counts[dateKey] = (counts[dateKey] || 0) + 1
        } catch (e) {
          console.warn('Invalid timestamp in anomaly:', anomaly.created_at || anomaly.timestamp, e)
        }
      })
    }
    
    // Convert to array with formatted dates (MM/DD)
    return Object.entries(counts)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([dateKey, count]) => {
        // Parse the dateKey (YYYY-MM-DD) and format as MM/DD
        const [year, month, day] = dateKey.split('-')
        return {
          date: `${month}/${day}`,
          count: count || 0
        }
      })
  }, [anomalies])

  const kpiCards = [
    {
      title: "MTBF",
      value: kpis?.mtbf?.toFixed(2) || "0.00",
      unit: "h",
      change: 0, // TODO: Calculate trend
      positive: true,
    },
    {
      title: "MTTR",
      value: kpis?.mttr?.toFixed(2) || "0.00",
      unit: "h",
      change: 0,
      positive: false,
    },
    {
      title: "OEE",
      value: kpis?.oee?.toFixed(2) || "0.00",
      unit: "%",
      change: 0,
      positive: true,
    },
    {
      title: "Availability",
      value: kpis?.availability?.toFixed(2) || "0.00",
      unit: "%",
      change: 0,
      positive: true,
    },
  ]

  return (
    <div className="space-y-8">
      {/* Page title */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
          <p className="text-muted-foreground mt-2">Real-time monitoring of your industrial assets</p>
        </div>
        <div className="flex items-center gap-2 text-sm">
          <div className={`h-2 w-2 rounded-full ${isConnected ? "bg-green-500 animate-pulse" : "bg-red-500"}`} />
          <span className="text-muted-foreground">{isConnected ? "Live" : "Offline"}</span>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {kpiCards.map((card) => (
          <Card key={card.title} className="border-border">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">{card.title}</CardTitle>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <Skeleton className="h-8 w-24" />
              ) : (
                <div className="flex items-end gap-4">
                  <div>
                    <p className="text-2xl font-bold">
                      {card.value}
                      <span className="text-sm text-muted-foreground ml-1">{card.unit}</span>
                    </p>
                  </div>
                  {card.change !== 0 && (
                    <div
                      className={`flex items-center gap-1 text-xs font-medium ${card.positive ? "text-green-600" : "text-red-600"}`}
                    >
                      {card.positive ? <ArrowUp className="h-3 w-3" /> : <ArrowDown className="h-3 w-3" />}
                      {Math.abs(card.change)}%
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Charts Grid */}
      <div className="grid gap-4 md:grid-cols-2">
        {/* Asset Status */}
        {isLoading ? (
          <Card className="border-border">
            <CardHeader>
              <CardTitle className="text-base">Asset Status Overview</CardTitle>
            </CardHeader>
            <CardContent>
              <Skeleton className="h-[300px] w-full" />
            </CardContent>
          </Card>
        ) : (
          <StatusPieChart data={statusData} title="Asset Status Overview" />
        )}

        {/* Anomaly Trend */}
        <Card className="border-border">
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <TrendingUp className="h-4 w-4" />
              Anomalies Trend
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={anomalyTrendData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                <XAxis dataKey="date" stroke="var(--color-muted-foreground)" />
                <YAxis stroke="var(--color-muted-foreground)" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "var(--color-card)",
                    border: "1px solid var(--color-border)",
                  }}
                />
                <Bar dataKey="count" fill="var(--color-chart-4)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* KPI Gauges */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {isLoading ? (
          <>
            <Skeleton className="h-[200px] w-full" />
            <Skeleton className="h-[200px] w-full" />
            <Skeleton className="h-[200px] w-full" />
            <Skeleton className="h-[200px] w-full" />
          </>
        ) : (
          <>
            <KPIGauge title="OEE" value={kpis?.oee || 0} unit="%" min={0} max={100} color="success" />
            <KPIGauge title="Availability" value={kpis?.availability || 0} unit="%" min={0} max={100} color="success" />
            <KPIGauge title="Reliability" value={kpis?.reliability || 0} unit="%" min={0} max={100} color="success" />
            <KPIGauge title="MTTR" value={kpis?.mttr || 0} unit="h" min={0} max={24} color="warning" />
          </>
        )}
      </div>

      {/* Recent Activity */}
      <div className="grid gap-4 md:grid-cols-2">
        {/* Recent Anomalies */}
        <Card className="border-border">
          <CardHeader>
            <CardTitle className="text-base">Recent Anomalies</CardTitle>
          </CardHeader>
          <CardContent>
            {/* Only show loading skeleton on initial load, not on refresh */}
            {recentAnomaliesLoading && (!recentAnomalies || recentAnomalies.length === 0) ? (
              <div className="space-y-3">
                {[1, 2, 3].map((i) => (
                  <Skeleton key={i} className="h-16 w-full" />
                ))}
              </div>
            ) : !recentAnomalies || recentAnomalies.length === 0 ? (
              <p className="text-sm text-muted-foreground text-center py-4">No recent anomalies</p>
            ) : (
              <div className="space-y-3">
                {recentAnomalies
                  .filter(anomaly => anomaly.is_anomaly === true) // Filter to only show actual anomalies
                  .sort((a, b) => {
                    // Sort by created_at (when detected) first, fall back to timestamp
                    const aTime = a.created_at ? new Date(a.created_at).getTime() : new Date(a.timestamp).getTime()
                    const bTime = b.created_at ? new Date(b.created_at).getTime() : new Date(b.timestamp).getTime()
                    return bTime - aTime // Newest first
                  })
                  .slice(0, 5)
                  .map((anomaly) => {
                    // Use created_at for display (when detected), fall back to timestamp
                    const displayTime = anomaly.created_at || anomaly.timestamp
                    return (
                      <div key={`${anomaly.id || anomaly.asset_id}-${displayTime}`} className="flex items-center justify-between p-3 border border-border rounded-lg">
                        <div>
                          <p className="font-medium text-sm">Asset {anomaly.asset_id}</p>
                          <p className="text-xs text-muted-foreground">
                            {new Date(displayTime).toLocaleTimeString('en-US', { 
                              hour: '2-digit', 
                              minute: '2-digit', 
                              second: '2-digit',
                              hour12: false 
                            })}
                          </p>
                        </div>
                        <div className={`px-3 py-1 rounded-full text-xs font-medium severity-${anomaly.severity}`}>
                          {anomaly.severity}
                        </div>
                      </div>
                    )
                  })}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Active Interventions */}
        <Card className="border-border">
          <CardHeader>
            <CardTitle className="text-base">Active Interventions</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="space-y-3">
                {[1, 2, 3].map((i) => (
                  <Skeleton key={i} className="h-16 w-full" />
                ))}
              </div>
            ) : !interventions || interventions.length === 0 ? (
              <p className="text-sm text-muted-foreground text-center py-4">No active interventions</p>
            ) : (
              <div className="space-y-3">
                {interventions.slice(0, 5).map((intervention, idx) => (
                <div key={idx} className="flex items-center justify-between p-3 border border-border rounded-lg">
                  <div>
                    <p className="font-medium text-sm">{intervention.title}</p>
                    <p className="text-xs text-muted-foreground">Asset {intervention.asset_id}</p>
                  </div>
                  <div
                    className={`px-3 py-1 rounded-full text-xs font-medium ${
                      intervention.status === "in_progress"
                        ? "bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-100"
                        : "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-100"
                    }`}
                  >
                    {intervention.status}
                  </div>
                </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
