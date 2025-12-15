import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8091/api/v1'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export const assetsApi = {
  getAll: (params?: any) => api.get('/assets', { params }),
  getById: (id: string) => api.get(`/assets/${id}`),
  getFeatures: (id: string, params?: any) => api.get(`/assets/${id}/features`, { params }),
}

export const anomaliesApi = {
  getAll: (params?: any) => api.get('/anomalies', { params }),
  getById: (id: string) => api.get(`/anomalies/${id}`),
}

export const rulApi = {
  getAll: (params?: any) => api.get('/rul', { params }),
  getLatest: (assetId: string) => api.get(`/rul/${assetId}/latest`),
}

export const interventionsApi = {
  getAll: (params?: any) => api.get('/interventions', { params }),
  getActive: () => api.get('/interventions/active'),
}

export const kpisApi = {
  getSummary: (params?: any) => api.get('/kpis/summary', { params }),
  getTrend: (metricName: string, params?: any) => api.get(`/kpis/trend/${metricName}`, { params }),
}

export const gisApi = {
  getAssetLocations: (params?: any) => api.get('/gis/assets', { params }),
  getAssetsWithinRadius: (params: any) => api.get('/gis/assets/within-radius', { params }),
  getFloorPlan: (floorPlanId?: string) => api.get('/gis/floor-plan', { params: { floor_plan_id: floorPlanId } }),
}

export const exportApi = {
  exportCSV: (params: any) => api.post('/export/csv', null, { params, responseType: 'blob' }),
  exportPDF: (params: any) => api.post('/export/pdf', null, { params, responseType: 'blob' }),
}

export const grafanaApi = {
  getDashboardUrl: (params?: any) => api.get('/grafana/dashboard-url', { params }),
}

export default api

