"use client"

import { useEffect, useState, useCallback } from "react"
import { apiClient } from "@/lib/api"
import type { RULPrediction } from "@/types"

export function useRUL(params?: {
  asset_id?: string
  sensor_id?: string
  start_date?: string
  end_date?: string
  limit?: number
  offset?: number
}) {
  const [predictions, setPredictions] = useState<RULPrediction[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [total, setTotal] = useState(0)

  const fetchRUL = useCallback(async () => {
    try {
      setIsLoading(true)
      const data = await apiClient.getRULPredictions({ ...params, limit: params?.limit || 100 })
      setPredictions(data.data)
      setTotal(data.total)
      setError(null)
    } catch (err) {
      console.error("Failed to fetch RUL predictions:", err)
      setError("Failed to load RUL predictions")
      setPredictions([])
    } finally {
      setIsLoading(false)
    }
  }, [params?.asset_id, params?.sensor_id, params?.start_date, params?.end_date, params?.limit, params?.offset])

  useEffect(() => {
    fetchRUL()
  }, [fetchRUL])

  return { predictions, isLoading, error, total, refetch: fetchRUL }
}

export function useLatestRUL(assetId: string) {
  const [prediction, setPrediction] = useState<RULPrediction | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchLatestRUL = useCallback(async () => {
    if (!assetId) {
      setIsLoading(false)
      return
    }

    try {
      setIsLoading(true)
      const data = await apiClient.getLatestRUL(assetId)
      setPrediction(data)
      setError(null)
    } catch (err) {
      console.error("Failed to fetch latest RUL:", err)
      setError("Failed to load latest RUL")
      setPrediction(null)
    } finally {
      setIsLoading(false)
    }
  }, [assetId])

  useEffect(() => {
    fetchLatestRUL()
  }, [fetchLatestRUL])

  return { prediction, isLoading, error, refetch: fetchLatestRUL }
}

