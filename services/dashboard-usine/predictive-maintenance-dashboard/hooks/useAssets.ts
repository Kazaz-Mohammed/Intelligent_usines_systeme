"use client"

import { useEffect, useState, useCallback } from "react"
import { apiClient } from "@/lib/api"
import type { Asset } from "@/types"

interface UseAssetsParams {
  page_size?: number
  status?: string
  asset_type?: string
}

export function useAssets(params: UseAssetsParams = {}) {
  const [assets, setAssets] = useState<Asset[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchAssets = useCallback(async () => {
    try {
      setIsLoading(true)
      console.log("[useAssets] Fetching assets with params:", params)
      const data = await apiClient.getAssets({ page_size: params.page_size || 100, ...params })
      console.log("[useAssets] Got response:", data)
      setAssets(data.assets || [])
      setError(null)
    } catch (err) {
      console.error("[useAssets] Failed to fetch assets:", err)
      setError("Failed to load assets")
      setAssets([])
    } finally {
      setIsLoading(false)
    }
  }, [params.page_size, params.status, params.asset_type])

  useEffect(() => {
    fetchAssets()
  }, [fetchAssets])

  return { assets, isLoading, error, refetch: fetchAssets }
}
