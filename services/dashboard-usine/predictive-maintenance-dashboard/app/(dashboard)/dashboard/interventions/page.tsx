"use client"

import { useState, useMemo } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { useInterventions } from "@/hooks/useInterventions"
import { Plus, Calendar, User } from "lucide-react"
import { Skeleton } from "@/components/ui/skeleton"

const statusColors = {
  planned: "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-100",
  in_progress: "bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-100",
  completed: "bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-100",
  cancelled: "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-100",
  overdue: "bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-100",
}

const priorityColors = {
  low: "bg-blue-50 text-blue-600",
  medium: "bg-yellow-50 text-yellow-600",
  high: "bg-orange-50 text-orange-600",
  urgent: "bg-red-50 text-red-600",
}

export default function InterventionsPage() {
  const [statusFilter, setStatusFilter] = useState<string>("all")
  const { interventions, isLoading, error } = useInterventions({
    status: statusFilter !== "all" ? statusFilter : undefined,
    limit: 100,
  })

  const handleNewIntervention = () => {
    // TODO: Open a modal/dialog to create a new intervention
    // For now, show an alert
    alert("New Intervention feature coming soon!\n\nThis will open a form to create a new maintenance work order.")
  }

  const filteredInterventions = useMemo(() => {
    if (!interventions || !Array.isArray(interventions)) return []
    if (statusFilter === "all") return interventions
    return interventions.filter((i) => i.status === statusFilter)
  }, [interventions, statusFilter])

  const statusStats = useMemo(() => {
    if (!interventions || !Array.isArray(interventions)) {
      return {
        total: 0,
        planned: 0,
        inProgress: 0,
        completed: 0,
        overdue: 0,
      }
    }
    return {
      total: interventions.length,
      planned: interventions.filter((i) => i.status === "planned").length,
      inProgress: interventions.filter((i) => i.status === "in_progress").length,
      completed: interventions.filter((i) => i.status === "completed").length,
      overdue: interventions.filter((i) => i.status === "overdue").length,
    }
  }, [interventions])

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Interventions</h1>
          <p className="text-muted-foreground mt-2">Manage maintenance work orders and interventions</p>
        </div>
        <Button 
          onClick={handleNewIntervention}
          className="gap-2 bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700"
        >
          <Plus className="h-4 w-4" />
          New Intervention
        </Button>
      </div>

      {error && (
        <div className="bg-destructive/10 text-destructive p-4 rounded-lg">
          {error}
        </div>
      )}

      {/* Status Summary */}
      <div className="grid gap-4 md:grid-cols-5">
        <Card className="border-border">
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Total</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? <Skeleton className="h-8 w-16" /> : <p className="text-3xl font-bold">{statusStats.total}</p>}
          </CardContent>
        </Card>
        <Card className="border-border">
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Planned</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? <Skeleton className="h-8 w-16" /> : <p className="text-3xl font-bold text-blue-600">{statusStats.planned}</p>}
          </CardContent>
        </Card>
        <Card className="border-border">
          <CardHeader className="pb-3">
            <CardTitle className="text-base">In Progress</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? <Skeleton className="h-8 w-16" /> : <p className="text-3xl font-bold text-purple-600">{statusStats.inProgress}</p>}
          </CardContent>
        </Card>
        <Card className="border-border">
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Completed</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? <Skeleton className="h-8 w-16" /> : <p className="text-3xl font-bold text-green-600">{statusStats.completed}</p>}
          </CardContent>
        </Card>
        <Card className="border-border">
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Overdue</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? <Skeleton className="h-8 w-16" /> : <p className="text-3xl font-bold text-red-600">{statusStats.overdue}</p>}
          </CardContent>
        </Card>
      </div>

      {/* Filter and Table */}
      <Card className="border-border">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Work Orders</CardTitle>
            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger className="w-40">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Status</SelectItem>
                <SelectItem value="planned">Planned</SelectItem>
                <SelectItem value="in_progress">In Progress</SelectItem>
                <SelectItem value="completed">Completed</SelectItem>
                <SelectItem value="overdue">Overdue</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardHeader>
        <CardContent>
          <div className="border border-border rounded-lg overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow className="border-border hover:bg-transparent">
                  <TableHead>Title</TableHead>
                  <TableHead>Asset</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Priority</TableHead>
                  <TableHead>Scheduled</TableHead>
                  <TableHead>Assigned To</TableHead>
                  <TableHead>Duration</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {isLoading ? (
                  <TableRow>
                    <TableCell colSpan={7} className="text-center py-8">
                      <Skeleton className="h-4 w-full" />
                    </TableCell>
                  </TableRow>
                ) : !filteredInterventions || filteredInterventions.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={7} className="text-center py-8">
                      <p className="text-muted-foreground">No interventions found</p>
                    </TableCell>
                  </TableRow>
                ) : (
                  filteredInterventions.map((intervention, idx) => (
                    <TableRow key={intervention.id || idx} className="border-border hover:bg-muted/50">
                      <TableCell className="font-medium">{intervention.title}</TableCell>
                      <TableCell>{intervention.asset_id}</TableCell>
                      <TableCell>
                        <Badge className={`${statusColors[intervention.status]} border-0`}>{intervention.status}</Badge>
                      </TableCell>
                      <TableCell>
                        <Badge className={`${priorityColors[intervention.priority]} border-0`}>
                          {intervention.priority}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-sm">
                        {intervention.scheduled_start ? (
                          <div className="flex items-center gap-2">
                            <Calendar className="h-3 w-3 text-muted-foreground" />
                            {new Date(intervention.scheduled_start).toLocaleDateString()}
                          </div>
                        ) : (
                          "-"
                        )}
                      </TableCell>
                      <TableCell className="text-sm">
                        {intervention.assigned_to ? (
                          <div className="flex items-center gap-2">
                            <User className="h-3 w-3 text-muted-foreground" />
                            {intervention.assigned_to}
                          </div>
                        ) : (
                          "-"
                        )}
                      </TableCell>
                      <TableCell className="text-sm">{intervention.estimated_duration ? `${intervention.estimated_duration}h` : "-"}</TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
