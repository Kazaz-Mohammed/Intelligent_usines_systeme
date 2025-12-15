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
  const { anomalies, isLoading: anomaliesLoading, refetch: refetchAnomalies } = useAnomalies({ limit: 10 })
  const { interventions, isLoading: interventionsLoading, refetch: refetchInterventions } = useInterventions({ limit: 10 })
  
  const isLoading = kpisLoading || assetsLoading || anomaliesLoading || interventionsLoading

  const { isConnected } = useWebSocket({
    onMessage: (message) => {
      console.log("Real-time update received:", message)
      // Refresh data on real-time updates
      if (message.type === "feature_update" || message.type === "anomaly_detected" || message.type === "rul_prediction") {
        refetchKPIs()
        refetchAssets()
        refetchAnomalies()
        refetchInterventions()
      }
    },
  })

  // Calculate status distribution from assets
  const statusData = useMemo(() => {
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

  // Generate anomaly trend from actual anomalies data
  const anomalyTrendData = useMemo(() => {
    const days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    const counts: Record<string, number> = {}
    
    // Initialize all days with 0
    days.forEach(day => counts[day] = 0)
    
    // Count anomalies per day of week
    anomalies.forEach(anomaly => {
      const date = new Date(anomaly.timestamp)
      const dayName = days[date.getDay()]
      counts[dayName] = (counts[dayName] || 0) + 1
    })
    
    // Return in Mon-Sun order
    return ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].map(day => ({
      date: day,
      count: counts[day] || 0
    }))
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
            {isLoading ? (
              <div className="space-y-3">
                {[1, 2, 3].map((i) => (
                  <Skeleton key={i} className="h-16 w-full" />
                ))}
              </div>
            ) : anomalies.length === 0 ? (
              <p className="text-sm text-muted-foreground text-center py-4">No recent anomalies</p>
            ) : (
              <div className="space-y-3">
                {anomalies.slice(0, 5).map((anomaly, idx) => (
                <div key={idx} className="flex items-center justify-between p-3 border border-border rounded-lg">
                  <div>
                    <p className="font-medium text-sm">Asset {anomaly.asset_id}</p>
                    <p className="text-xs text-muted-foreground">{new Date(anomaly.timestamp).toLocaleTimeString()}</p>
                  </div>
                  <div className={`px-3 py-1 rounded-full text-xs font-medium severity-${anomaly.severity}`}>
                    {anomaly.severity}
                  </div>
                </div>
                ))}
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
            ) : interventions.length === 0 ? (
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
