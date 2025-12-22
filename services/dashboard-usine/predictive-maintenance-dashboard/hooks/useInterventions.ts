"use client"

import { useEffect, useState, useCallback } from "react"
import { apiClient } from "@/lib/api"
import type { Intervention } from "@/types"

export function useInterventions(params?: {
  asset_id?: string
  status?: string
  limit?: number
  offset?: number
}) {
  const [interventions, setInterventions] = useState<Intervention[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [total, setTotal] = useState(0)

  const fetchInterventions = useCallback(async () => {
    try {
      setIsLoading(true)
      const data = await apiClient.getInterventions({ ...params, limit: params?.limit || 100 })
      setInterventions(data.interventions || [])
      setTotal(data.total || 0)
      setError(null)
    } catch (err) {
      console.error("Failed to fetch interventions:", err)
      setError("Failed to load interventions")
      setInterventions([])
    } finally {
      setIsLoading(false)
    }
  }, [params?.asset_id, params?.status, params?.limit, params?.offset])

  useEffect(() => {
    fetchInterventions()
  }, [fetchInterventions])

  return { interventions, isLoading, error, total, refetch: fetchInterventions }
}

