import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { Asset, Anomaly, RULPrediction, Intervention, KPISummary } from '../types'
import { assetsApi, anomaliesApi, rulApi, interventionsApi, kpisApi } from '../services/api'
import { useWebSocket } from '../hooks/useWebSocket'

interface DashboardContextType {
  assets: Asset[]
  anomalies: Anomaly[]
  rulPredictions: RULPrediction[]
  interventions: Intervention[]
  kpiSummary: KPISummary | null
  loading: boolean
  error: string | null
  refreshAssets: () => Promise<void>
  refreshAnomalies: () => Promise<void>
  refreshRUL: () => Promise<void>
  refreshInterventions: () => Promise<void>
  refreshKPIs: () => Promise<void>
  isConnected: boolean
}

const DashboardContext = createContext<DashboardContextType | undefined>(undefined)

export const DashboardProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [assets, setAssets] = useState<Asset[]>([])
  const [anomalies, setAnomalies] = useState<Anomaly[]>([])
  const [rulPredictions, setRulPredictions] = useState<RULPrediction[]>([])
  const [interventions, setInterventions] = useState<Intervention[]>([])
  const [kpiSummary, setKpiSummary] = useState<KPISummary | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const refreshAssets = async () => {
    try {
      const response = await assetsApi.getAll()
      setAssets(response.data.assets || [])
    } catch (err: any) {
      setError(err.message)
    }
  }

  const refreshAnomalies = async () => {
    try {
      const response = await anomaliesApi.getAll({ limit: 100 })
      setAnomalies(response.data.anomalies || [])
    } catch (err: any) {
      setError(err.message)
    }
  }

  const refreshRUL = async () => {
    try {
      const response = await rulApi.getAll({ limit: 100 })
      setRulPredictions(response.data.predictions || [])
    } catch (err: any) {
      setError(err.message)
    }
  }

  const refreshInterventions = async () => {
    try {
      const response = await interventionsApi.getAll({ limit: 100 })
      setInterventions(response.data.interventions || [])
    } catch (err: any) {
      setError(err.message)
    }
  }

  const refreshKPIs = async () => {
    try {
      const response = await kpisApi.getSummary()
      setKpiSummary(response.data)
    } catch (err: any) {
      setError(err.message)
    }
  }

  // WebSocket for real-time updates
  const { isConnected } = useWebSocket((message) => {
    switch (message.type) {
      case 'feature_update':
        refreshAssets()
        break
      case 'anomaly_detected':
        refreshAnomalies()
        break
      case 'rul_prediction':
        refreshRUL()
        break
    }
  })

  // Initial load
  useEffect(() => {
    const loadAll = async () => {
      setLoading(true)
      try {
        await Promise.all([
          refreshAssets(),
          refreshAnomalies(),
          refreshRUL(),
          refreshInterventions(),
          refreshKPIs(),
        ])
      } catch (err: any) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    loadAll()
  }, [])

  // Auto-refresh every 30 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      refreshAssets()
      refreshAnomalies()
      refreshRUL()
      refreshInterventions()
      refreshKPIs()
    }, 30000)

    return () => clearInterval(interval)
  }, [])

  return (
    <DashboardContext.Provider
      value={{
        assets,
        anomalies,
        rulPredictions,
        interventions,
        kpiSummary,
        loading,
        error,
        refreshAssets,
        refreshAnomalies,
        refreshRUL,
        refreshInterventions,
        refreshKPIs,
        isConnected,
      }}
    >
      {children}
    </DashboardContext.Provider>
  )
}

export const useDashboard = () => {
  const context = useContext(DashboardContext)
  if (!context) {
    throw new Error('useDashboard must be used within DashboardProvider')
  }
  return context
}

