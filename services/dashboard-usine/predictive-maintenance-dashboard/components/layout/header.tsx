"use client"

import { Bell, Search, Settings, Moon, Sun, AlertTriangle, TrendingDown, X, User, LogOut, Info, RefreshCw, Server } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { useEffect, useState } from "react"
import { useWebSocket } from "@/hooks/useWebSocket"
import { useAnomalies } from "@/hooks/useAnomalies"
import { useRUL } from "@/hooks/useRUL"
import { useAuth } from "@/context/auth-context"
import { useRouter } from "next/navigation"
import type { WebSocketMessage } from "@/types"
import { Badge } from "@/components/ui/badge"
import Link from "next/link"

interface NotificationItem {
  id: string
  type: "anomaly" | "rul"
  title: string
  message: string
  timestamp: string
  severity?: string
  assetId: string
  link: string
}

export function Header() {
  const [isDark, setIsDark] = useState(false)
  const [notifications, setNotifications] = useState(0)
  const [isNotificationOpen, setIsNotificationOpen] = useState(false)
  const [notificationItems, setNotificationItems] = useState<NotificationItem[]>([])
  const { user, logout } = useAuth()
  const router = useRouter()
  
  const { isConnected } = useWebSocket({
    onMessage: (message: WebSocketMessage) => {
      if (message.type === "anomaly_detected" || message.type === "rul_prediction") {
        setNotifications((prev) => prev + 1)
      }
    },
  })

  // Fetch recent anomalies and RUL predictions for notifications
  const { anomalies } = useAnomalies({ limit: 10, is_anomaly: true })
  const { predictions } = useRUL({ limit: 10 })

  useEffect(() => {
    const isDarkMode = document.documentElement.classList.contains("dark")
    setIsDark(isDarkMode)
  }, [])

  // Build notification items from anomalies and RUL predictions
  useEffect(() => {
    const items: NotificationItem[] = []
    
    // Add recent anomalies (last 24 hours)
    const oneDayAgo = new Date()
    oneDayAgo.setDate(oneDayAgo.getDate() - 1)
    
    // Guard against undefined anomalies
    if (anomalies && Array.isArray(anomalies)) {
      anomalies
        .filter((a) => {
          const timestamp = new Date(a.timestamp)
          return timestamp > oneDayAgo && a.is_anomaly
        })
        .slice(0, 5)
        .forEach((anomaly) => {
          items.push({
            id: `anomaly-${anomaly.id || anomaly.asset_id}-${anomaly.timestamp}`,
            type: "anomaly",
            title: `Anomaly Detected: ${anomaly.asset_id}`,
            message: `Criticality: ${anomaly.severity || anomaly.criticality || "unknown"}`,
            timestamp: new Date(anomaly.timestamp).toLocaleString(),
            severity: anomaly.severity || anomaly.criticality,
            assetId: anomaly.asset_id,
            link: `/dashboard/anomalies?asset_id=${anomaly.asset_id}`,
          })
        })
    }
    
    // Add low RUL predictions (< 180 hours)
    // Guard against undefined predictions
    if (predictions && Array.isArray(predictions)) {
      predictions
        .filter((p) => p.rul_prediction < 180)
        .slice(0, 5)
        .forEach((pred) => {
          items.push({
            id: `rul-${pred.id || pred.asset_id}-${pred.timestamp}`,
            type: "rul",
            title: `Low RUL: ${pred.asset_id}`,
            message: `RUL: ${pred.rul_prediction.toFixed(0)} hours remaining`,
            timestamp: new Date(pred.timestamp).toLocaleString(),
            assetId: pred.asset_id,
            link: `/dashboard/rul?asset_id=${pred.asset_id}`,
          })
        })
    }
    
    // Sort by timestamp (newest first)
    items.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
    setNotificationItems(items.slice(0, 10))
    setNotifications(items.length)
  }, [anomalies, predictions])

  const toggleDarkMode = () => {
    document.documentElement.classList.toggle("dark")
    setIsDark(!isDark)
  }

  const handleNotificationOpen = (open: boolean) => {
    setIsNotificationOpen(open)
    if (open) {
      // Clear notifications when opened
      setNotifications(0)
    }
  }

  const handleLogout = () => {
    logout()
    router.push("/login")
  }

  const handleRefresh = () => {
    window.location.reload()
  }

  return (
    <header className="sticky top-0 z-30 flex h-16 items-center justify-between gap-4 border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 px-6 lg:ml-64">
      {/* Search bar */}
      <div className="hidden md:flex flex-1 max-w-sm">
        <div className="relative w-full">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input placeholder="Search assets..." className="pl-10" />
        </div>
      </div>

      {/* Right section */}
      <div className="flex items-center gap-2 ml-auto">
        {/* Connection status */}
        <div className="hidden sm:flex items-center gap-2 text-xs">
          <div className={`h-2 w-2 rounded-full ${isConnected ? "bg-green-500" : "bg-red-500"}`} />
          <span className="text-muted-foreground">{isConnected ? "Connected" : "Offline"}</span>
        </div>

        <Button variant="ghost" size="icon" onClick={toggleDarkMode} className="h-9 w-9">
          {isDark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </Button>

        <Popover open={isNotificationOpen} onOpenChange={handleNotificationOpen}>
          <PopoverTrigger asChild>
            <Button variant="ghost" size="icon" className="h-9 w-9 relative">
              <Bell className="h-4 w-4" />
              {notifications > 0 && (
                <span className="absolute top-1 right-1 h-5 w-5 bg-red-500 rounded-full text-white text-xs flex items-center justify-center font-bold">
                  {notifications > 9 ? "9+" : notifications}
                </span>
              )}
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-80 p-0" align="end">
            <div className="flex items-center justify-between p-4 border-b">
              <h3 className="font-semibold text-sm">Notifications</h3>
              {notificationItems.length > 0 && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-6 px-2 text-xs"
                  onClick={() => {
                    setNotificationItems([])
                    setNotifications(0)
                  }}
                >
                  Clear all
                </Button>
              )}
            </div>
            <div className="max-h-96 overflow-y-auto">
              {notificationItems.length === 0 ? (
                <div className="p-8 text-center text-sm text-muted-foreground">
                  <Bell className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p>No new notifications</p>
                </div>
              ) : (
                <div className="divide-y">
                  {notificationItems.map((item) => (
                    <Link
                      key={item.id}
                      href={item.link}
                      onClick={() => handleNotificationOpen(false)}
                      className="block p-4 hover:bg-muted/50 transition-colors"
                    >
                      <div className="flex items-start gap-3">
                        <div className={`mt-1 p-2 rounded-full ${
                          item.type === "anomaly"
                            ? "bg-red-100 dark:bg-red-900/20"
                            : "bg-orange-100 dark:bg-orange-900/20"
                        }`}>
                          {item.type === "anomaly" ? (
                            <AlertTriangle className="h-4 w-4 text-red-600 dark:text-red-400" />
                          ) : (
                            <TrendingDown className="h-4 w-4 text-orange-600 dark:text-orange-400" />
                          )}
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-start justify-between gap-2">
                            <p className="text-sm font-medium truncate">{item.title}</p>
                            {item.severity && (
                              <Badge
                                variant="outline"
                                className={`text-xs ${
                                  item.severity === "critical"
                                    ? "border-red-500 text-red-600"
                                    : item.severity === "high"
                                    ? "border-orange-500 text-orange-600"
                                    : "border-yellow-500 text-yellow-600"
                                }`}
                              >
                                {item.severity}
                              </Badge>
                            )}
                          </div>
                          <p className="text-xs text-muted-foreground mt-1">{item.message}</p>
                          <p className="text-xs text-muted-foreground mt-1">{item.timestamp}</p>
                        </div>
                      </div>
                    </Link>
                  ))}
                </div>
              )}
            </div>
            {notificationItems.length > 0 && (
              <div className="p-2 border-t">
                <Link
                  href="/dashboard/anomalies"
                  onClick={() => handleNotificationOpen(false)}
                  className="block text-center text-xs text-primary hover:underline py-2"
                >
                  View all notifications
                </Link>
              </div>
            )}
          </PopoverContent>
        </Popover>

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon" className="h-9 w-9">
              <Settings className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-56">
            <DropdownMenuLabel>
              <div className="flex flex-col space-y-1">
                <p className="text-sm font-medium leading-none">Settings</p>
                {user && (
                  <p className="text-xs leading-none text-muted-foreground">
                    {user.email || "User"}
                  </p>
                )}
              </div>
            </DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={() => router.push("/dashboard/profile")}>
              <User className="mr-2 h-4 w-4" />
              <span>Profile</span>
            </DropdownMenuItem>
            <DropdownMenuItem onClick={handleRefresh}>
              <RefreshCw className="mr-2 h-4 w-4" />
              <span>Refresh Page</span>
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem disabled>
              <Server className="mr-2 h-4 w-4" />
              <div className="flex flex-col">
                <span>API Status</span>
                <span className="text-xs text-muted-foreground">
                  {isConnected ? "Connected" : "Disconnected"}
                </span>
              </div>
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={() => router.push("/dashboard/about")}>
              <Info className="mr-2 h-4 w-4" />
              <span>About</span>
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={handleLogout} variant="destructive">
              <LogOut className="mr-2 h-4 w-4" />
              <span>Logout</span>
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </header>
  )
}
