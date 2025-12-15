import React from 'react'
import Plotly from 'plotly.js'
import createPlotlyComponent from 'react-plotly.js/factory'
import './RULGauge.css'

const Plot = createPlotlyComponent(Plotly)

interface RULGaugeProps {
  value: number
  assetName?: string
  maxValue?: number
}

const RULGauge: React.FC<RULGaugeProps> = ({ value, assetName, maxValue = 1000 }) => {
  const percentage = (value / maxValue) * 100
  const getColor = () => {
    if (percentage > 50) return '#27ae60'
    if (percentage > 25) return '#f39c12'
    return '#e74c3c'
  }

  const data: any = [
    {
      type: 'indicator',
      mode: 'gauge+number',
      value: value,
      domain: { x: [0, 1], y: [0, 1] },
      title: { text: assetName || 'RUL (heures)', font: { size: 16 } },
      gauge: {
        axis: { range: [null, maxValue] },
        bar: { color: getColor() },
        steps: [
          { range: [0, maxValue * 0.25], color: '#e74c3c' },
          { range: [maxValue * 0.25, maxValue * 0.5], color: '#f39c12' },
          { range: [maxValue * 0.5, maxValue], color: '#27ae60' },
        ],
        threshold: {
          line: { color: 'red', width: 4 },
          thickness: 0.75,
          value: maxValue * 0.1,
        },
      },
    },
  ]

  const layout: any = {
    width: 400,
    height: 300,
    margin: { t: 0, b: 0, l: 0, r: 0 },
    paper_bgcolor: 'white',
    plot_bgcolor: 'white',
  }

  return (
    <div className="rul-gauge">
      <Plot data={data} layout={layout} config={{ displayModeBar: false }} useResizeHandler={true} />
    </div>
  )
}

export default RULGauge

