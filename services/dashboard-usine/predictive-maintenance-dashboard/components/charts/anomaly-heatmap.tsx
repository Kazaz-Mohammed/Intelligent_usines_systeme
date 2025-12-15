"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

interface AnomalyHeatmapData {
  asset: string
  low: number
  medium: number
  high: number
  critical: number
}

interface AnomalyHeatmapProps {
  data: AnomalyHeatmapData[]
}

export function AnomalyHeatmap({ data }: AnomalyHeatmapProps) {
  const getColor = (value: number, max: number) => {
    const ratio = value / Math.max(max, 1)
    if (ratio === 0) return "bg-gray-100 dark:bg-gray-800"
    if (ratio < 0.25) return "bg-blue-100 dark:bg-blue-900"
    if (ratio < 0.5) return "bg-yellow-100 dark:bg-yellow-900"
    if (ratio < 0.75) return "bg-orange-100 dark:bg-orange-900"
    return "bg-red-100 dark:bg-red-900"
  }

  const maxValue = Math.max(...data.flatMap((d) => [d.low, d.medium, d.high, d.critical]))

  return (
    <Card className="border-border">
      <CardHeader>
        <CardTitle className="text-base">Anomaly Distribution Heatmap</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <div className="space-y-2 min-w-max">
            {data.map((row) => (
              <div key={row.asset} className="flex items-center gap-4">
                <div className="w-24 text-sm font-medium truncate">{row.asset}</div>
                <div className="flex gap-1">
                  {[row.low, row.medium, row.high, row.critical].map((value, idx) => (
                    <div
                      key={idx}
                      className={`w-12 h-12 flex items-center justify-center text-xs font-medium rounded border border-border ${getColor(value, maxValue)}`}
                      title={`Severity ${["Low", "Medium", "High", "Critical"][idx]}: ${value}`}
                    >
                      {value}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
        <div className="flex gap-4 mt-6 justify-center flex-wrap text-xs">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-blue-100 dark:bg-blue-900 rounded" />
            <span>Low</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-yellow-100 dark:bg-yellow-900 rounded" />
            <span>Medium</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-orange-100 dark:bg-orange-900 rounded" />
            <span>High</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-red-100 dark:bg-red-900 rounded" />
            <span>Critical</span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
