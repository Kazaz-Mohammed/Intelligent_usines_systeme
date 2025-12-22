// Core data types for the predictive maintenance system

export interface User {
  id: string
  name: string
  email: string
  role: "admin" | "operator" | "viewer"
}

export interface AuthToken {
  token: string
  user: User
}

export interface Asset {
  id: string
  name: string
  type: string
  status: "operational" | "warning" | "critical" | "maintenance" | "offline"
  location?: string
  coordinates?: { lat: number; lon: number }
  metadata?: Record<string, any>
  created_at?: string
  updated_at?: string
}

export interface AssetDetail extends Asset {
  current_rul?: number
  anomaly_count: number
  last_anomaly?: string
  last_maintenance?: string
  next_maintenance?: string
  features?: any[]
}

export interface Anomaly {
  id?: string
  asset_id: string
  sensor_id?: string
  timestamp: string
  created_at?: string  // When the anomaly was detected/saved
  severity: "low" | "medium" | "warning" | "high" | "critical"
  is_anomaly: boolean
  anomaly_score?: number
  model_used?: string
  features?: Record<string, any>
  metadata?: Record<string, any>
}

export interface RULPrediction {
  id?: string
  asset_id: string
  sensor_id?: string
  timestamp: string
  rul_prediction: number
  confidence_interval_lower?: number
  confidence_interval_upper?: number
  confidence_level?: number
  uncertainty?: number
  model_used?: string
  model_scores?: Record<string, number>
  features?: Record<string, any>
  metadata?: Record<string, any>
}

export interface Intervention {
  id?: string
  asset_id: string
  work_order_id?: string
  title: string
  description?: string
  status: "planned" | "scheduled" | "in_progress" | "completed" | "cancelled" | "overdue"
  priority: "low" | "medium" | "high" | "urgent"
  scheduled_start?: string
  scheduled_end?: string
  actual_start?: string
  actual_end?: string
  assigned_to?: string
  estimated_duration?: number
  actual_duration?: number
  cost?: number
  metadata?: Record<string, any>
  created_at?: string
  updated_at?: string
}

export interface KPIMetrics {
  mtbf?: number
  mttr?: number
  oee?: number
  availability?: number
  reliability?: number
  timestamp: string
}

export interface KPITrend {
  metric_name: string
  values: number[]
  timestamps: string[]
  unit?: string
}

export interface PaginatedResponse<T> {
  data: T[]
  total: number
  page: number
  page_size: number
}

export interface WebSocketMessage {
  type: "feature_update" | "anomaly_detected" | "rul_prediction" | "pong" | "connection"
  asset_id?: string
  data?: any
  timestamp: string
}
