"use client"

import { useEffect, useState, useCallback } from "react"
import { apiClient } from "@/lib/api"
import type { KPIMetrics, KPITrend } from "@/types"

export function useKPISummary(params?: { asset_id?: string; days?: number }) {
  const [kpis, setKpis] = useState<KPIMetrics | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchKPIs = useCallback(async () => {
    try {
      setIsLoading(true)
      const data = await apiClient.getKPISummary({ ...params, days: params?.days || 30 })
      setKpis(data)
      setError(null)
    } catch (err) {
      console.error("Failed to fetch KPI summary:", err)
      setError("Failed to load KPI summary")
      setKpis(null)
    } finally {
      setIsLoading(false)
    }
  }, [params?.asset_id, params?.days])

  useEffect(() => {
    fetchKPIs()
  }, [fetchKPIs])

  return { kpis, isLoading, error, refetch: fetchKPIs }
}

export function useKPITrend(metricName: string, params?: { asset_id?: string; days?: number }) {
  const [trend, setTrend] = useState<KPITrend | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchTrend = useCallback(async () => {
    if (!metricName) {
      setIsLoading(false)
      return
    }

    try {
      setIsLoading(true)
      const data = await apiClient.getKPITrend(metricName, { ...params, days: params?.days || 30 })
      setTrend(data)
      setError(null)
    } catch (err) {
      console.error("Failed to fetch KPI trend:", err)
      setError("Failed to load KPI trend")
      setTrend(null)
    } finally {
      setIsLoading(false)
    }
  }, [metricName, params?.asset_id, params?.days])

  useEffect(() => {
    fetchTrend()
  }, [fetchTrend])

  return { trend, isLoading, error, refetch: fetchTrend }
}

