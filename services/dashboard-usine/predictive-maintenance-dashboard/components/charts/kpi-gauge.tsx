"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

interface KPIGaugeProps {
  title: string
  value: number
  unit: string
  min?: number
  max?: number
  color?: "success" | "warning" | "critical"
}

export function KPIGauge({ title, value, unit, min = 0, max = 100, color = "success" }: KPIGaugeProps) {
  const percentage = ((value - min) / (max - min)) * 100
  const clampedPercentage = Math.min(Math.max(percentage, 0), 100)

  const colorMap = {
    success: "from-green-500 to-green-600",
    warning: "from-yellow-500 to-yellow-600",
    critical: "from-red-500 to-red-600",
  }

  return (
    <Card className="border-border">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col items-center">
          <div className="relative w-32 h-32 flex items-center justify-center">
            <svg className="w-full h-full transform -rotate-90" viewBox="0 0 120 120">
              {/* Background circle */}
              <circle cx="60" cy="60" r="50" fill="none" stroke="var(--color-border)" strokeWidth="8" />
              {/* Progress circle */}
              <circle
                cx="60"
                cy="60"
                r="50"
                fill="none"
                stroke={`url(#gradient-${color})`}
                strokeWidth="8"
                strokeDasharray={`${(clampedPercentage / 100) * 314} 314`}
                strokeLinecap="round"
              />
              <defs>
                <linearGradient id={`gradient-${color}`} x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="var(--color-chart-1)" />
                  <stop offset="100%" stopColor="var(--color-chart-4)" />
                </linearGradient>
              </defs>
            </svg>
            <div className="absolute text-center">
              <p className="text-2xl font-bold">{value}</p>
              <p className="text-xs text-muted-foreground">{unit}</p>
            </div>
          </div>
          <p className="text-xs text-muted-foreground mt-4">{clampedPercentage.toFixed(0)}% of target</p>
        </div>
      </CardContent>
    </Card>
  )
}
