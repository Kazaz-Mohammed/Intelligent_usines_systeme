import React from 'react'
import Plotly from 'plotly.js'
import createPlotlyComponent from 'react-plotly.js/factory'
import { Anomaly } from '../types'

const Plot = createPlotlyComponent(Plotly)

interface AnomalyHeatmapProps {
  anomalies: Anomaly[]
}

const AnomalyHeatmap: React.FC<AnomalyHeatmapProps> = ({ anomalies }) => {
  // Group anomalies by asset and severity
  const assetSeverityMap: Record<string, Record<string, number>> = {}
  
  anomalies.forEach(anomaly => {
    if (!assetSeverityMap[anomaly.asset_id]) {
      assetSeverityMap[anomaly.asset_id] = { low: 0, medium: 0, high: 0, critical: 0 }
    }
    assetSeverityMap[anomaly.asset_id][anomaly.severity] = 
      (assetSeverityMap[anomaly.asset_id][anomaly.severity] || 0) + 1
  })

  const assets = Object.keys(assetSeverityMap)
  const severities = ['low', 'medium', 'high', 'critical']
  
  const z = severities.map(severity => 
    assets.map(asset => assetSeverityMap[asset][severity] || 0)
  )

  const plotData: any = [
    {
      z: z,
      x: assets,
      y: severities,
      type: 'heatmap',
      colorscale: [
        [0, '#27ae60'],
        [0.33, '#f39c12'],
        [0.66, '#e67e22'],
        [1, '#e74c3c'],
      ],
      showscale: true,
    },
  ]

  const layout: any = {
    title: 'Distribution des Anomalies par Actif et Sévérité',
    xaxis: { title: 'Actifs' },
    yaxis: { title: 'Sévérité' },
    height: 400,
    margin: { t: 50, b: 100, l: 100, r: 20 },
    paper_bgcolor: 'white',
    plot_bgcolor: 'white',
  }

  return <Plot data={plotData} layout={layout} config={{ displayModeBar: true }} />
}

export default AnomalyHeatmap

