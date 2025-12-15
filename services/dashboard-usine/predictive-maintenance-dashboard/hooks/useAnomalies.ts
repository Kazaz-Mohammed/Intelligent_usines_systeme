"use client"

import { useEffect, useState, useCallback } from "react"
import { apiClient } from "@/lib/api"
import type { Anomaly } from "@/types"

export function useAnomalies(params?: {
  asset_id?: string
  sensor_id?: string
  start_date?: string
  end_date?: string
  criticality?: string
  is_anomaly?: boolean
  limit?: number
  offset?: number
}) {
  const [anomalies, setAnomalies] = useState<Anomaly[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [total, setTotal] = useState(0)

  const fetchAnomalies = useCallback(async () => {
    try {
      setIsLoading(true)
      const data = await apiClient.getAnomalies({ ...params, limit: params?.limit || 100 })
      setAnomalies(data.data)
      setTotal(data.total)
      setError(null)
    } catch (err) {
      console.error("Failed to fetch anomalies:", err)
      setError("Failed to load anomalies")
      setAnomalies([])
    } finally {
      setIsLoading(false)
    }
  }, [params?.asset_id, params?.sensor_id, params?.start_date, params?.end_date, params?.criticality, params?.is_anomaly, params?.limit, params?.offset])

  useEffect(() => {
    fetchAnomalies()
  }, [fetchAnomalies])

  return { anomalies, isLoading, error, total, refetch: fetchAnomalies }
}

