import React from 'react'
import Plotly from 'plotly.js'
import createPlotlyComponent from 'react-plotly.js/factory'

const Plot = createPlotlyComponent(Plotly)

interface TimeSeriesChartProps {
  data: Array<{ timestamp: string; value: number }>
  title?: string
  yAxisLabel?: string
}

const TimeSeriesChart: React.FC<TimeSeriesChartProps> = ({ data, title, yAxisLabel = 'Value' }) => {
  const plotData: any = [
    {
      x: data.map(d => d.timestamp),
      y: data.map(d => d.value),
      type: 'scatter',
      mode: 'lines+markers',
      name: yAxisLabel,
      line: { color: '#3498db' },
    },
  ]

  const layout: any = {
    title: title || 'Time Series',
    xaxis: { title: 'Time' },
    yaxis: { title: yAxisLabel },
    height: 400,
    margin: { t: 50, b: 50, l: 60, r: 20 },
    paper_bgcolor: 'white',
    plot_bgcolor: 'white',
  }

  return <Plot data={plotData} layout={layout} config={{ displayModeBar: true }} />
}

export default TimeSeriesChart

