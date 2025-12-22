"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Input } from "@/components/ui/input"
import { Download, Filter } from "lucide-react"
import { apiClient } from "@/lib/api"
import { useAssets } from "@/hooks/useAssets"
import { toast } from "sonner"

export default function ReportsPage() {
  const [startDate, setStartDate] = useState(new Date(Date.now() - 2592000000).toISOString().split("T")[0])
  const [endDate, setEndDate] = useState(new Date().toISOString().split("T")[0])
  const [dataType, setDataType] = useState("features")
  const [assetFilter, setAssetFilter] = useState<string>("all")
  const [isExporting, setIsExporting] = useState(false)
  const { assets } = useAssets({ page_size: 100 })

  const handleExportCSV = async () => {
    try {
      setIsExporting(true)
      const blob = await apiClient.exportCSV({
        asset_id: assetFilter !== "all" ? assetFilter : undefined,
        start_date: startDate,
        end_date: endDate,
        data_type: dataType,
      })
      
      // Create download link
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `report_${dataType}_${startDate}_${endDate}.csv`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
      
      toast.success("CSV exported successfully")
    } catch (error) {
      console.error("Export failed:", error)
      toast.error("Failed to export CSV")
    } finally {
      setIsExporting(false)
    }
  }

  const handleExportPDF = async () => {
    try {
      setIsExporting(true)
      const blob = await apiClient.exportPDF({
        asset_id: assetFilter !== "all" ? assetFilter : undefined,
        start_date: startDate,
        end_date: endDate,
      })
      
      // Create download link
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `report_${startDate}_${endDate}.pdf`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
      
      toast.success("PDF exported successfully")
    } catch (error) {
      console.error("Export failed:", error)
      toast.error("Failed to export PDF")
    } finally {
      setIsExporting(false)
    }
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Reports & Export</h1>
        <p className="text-muted-foreground mt-2">Generate and export reports for your maintenance data</p>
      </div>

      {/* Report Builder */}
      <Card className="border-border">
        <CardHeader>
          <CardTitle>Generate Report</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* Filters */}
            <div className="grid gap-4 md:grid-cols-2">
              <div>
                <label className="text-sm font-medium mb-2 block">Start Date</label>
                <Input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} />
              </div>
              <div>
                <label className="text-sm font-medium mb-2 block">End Date</label>
                <Input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} />
              </div>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <div>
                <label className="text-sm font-medium mb-2 block">Asset</label>
                <Select value={assetFilter} onValueChange={setAssetFilter}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Assets</SelectItem>
                    {assets.map((asset) => (
                      <SelectItem key={asset.id} value={asset.id}>
                        {asset.name} ({asset.id})
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div>
                <label className="text-sm font-medium mb-2 block">Data Type</label>
                <Select value={dataType} onValueChange={setDataType}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Data</SelectItem>
                    <SelectItem value="features">Features</SelectItem>
                    <SelectItem value="anomalies">Anomalies</SelectItem>
                    <SelectItem value="rul">RUL Predictions</SelectItem>
                    <SelectItem value="interventions">Interventions</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            {/* Export Buttons */}
            <div className="flex gap-3 pt-4">
              <Button
                onClick={handleExportCSV}
                variant="outline"
                className="gap-2 bg-transparent"
                disabled={isExporting}
              >
                <Download className="h-4 w-4" />
                {isExporting ? "Exporting..." : "Export CSV"}
              </Button>
              <Button
                onClick={handleExportPDF}
                className="gap-2 bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700"
                disabled={isExporting}
              >
                <Download className="h-4 w-4" />
                {isExporting ? "Exporting..." : "Export PDF"}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Report Templates */}
      <div className="grid gap-4 md:grid-cols-2">
        {[
          {
            title: "Maintenance Summary",
            description: "Overview of completed and planned maintenance activities",
            dataType: "interventions",
            reportType: "summary",
          },
          {
            title: "Anomaly Report",
            description: "Detailed anomaly detection and analysis",
            dataType: "anomalies",
            reportType: "detailed",
          },
          {
            title: "RUL Predictions",
            description: "Remaining Useful Life predictions and trends",
            dataType: "rul",
            reportType: "detailed",
          },
          {
            title: "Equipment Performance",
            description: "Overall equipment effectiveness and KPI metrics",
            dataType: "all",
            reportType: "kpi",
          },
        ].map((report) => {
          const handleGenerate = async () => {
            try {
              setIsExporting(true)
              // Set the data type for CSV export
              setDataType(report.dataType)
              
              // Generate PDF report
              const blob = await apiClient.exportPDF({
                asset_id: assetFilter !== "all" ? assetFilter : undefined,
                start_date: startDate,
                end_date: endDate,
                report_type: report.reportType,
              })
              
              // Create download link
              const url = window.URL.createObjectURL(blob)
              const a = document.createElement("a")
              a.href = url
              a.download = `${report.title.toLowerCase().replace(/\s+/g, "_")}_${startDate}_${endDate}.pdf`
              document.body.appendChild(a)
              a.click()
              window.URL.revokeObjectURL(url)
              document.body.removeChild(a)
              
              toast.success(`${report.title} generated successfully`)
            } catch (error) {
              console.error("Report generation failed:", error)
              toast.error(`Failed to generate ${report.title}`)
            } finally {
              setIsExporting(false)
            }
          }

          return (
            <Card
              key={report.title}
              className="border-border cursor-pointer hover:border-primary hover:shadow-lg transition-all"
            >
              <CardHeader>
                <CardTitle className="text-base">{report.title}</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-4">{report.description}</p>
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="gap-2 bg-transparent"
                  onClick={handleGenerate}
                  disabled={isExporting}
                >
                  <Filter className="h-3 w-3" />
                  {isExporting ? "Generating..." : "Generate"}
                </Button>
              </CardContent>
            </Card>
          )
        })}
      </div>
    </div>
  )
}
