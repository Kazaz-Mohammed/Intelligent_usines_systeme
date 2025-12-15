"use client"

import Link from "next/link"
import { usePathname, useRouter } from "next/navigation"
import { useAuth } from "@/context/auth-context"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import {
  LayoutDashboard,
  Package,
  AlertTriangle,
  TrendingUp,
  Wrench,
  Map,
  FileText,
  LogOut,
  Menu,
  X,
} from "lucide-react"
import { useState } from "react"

const sidebarLinks = [
  {
    href: "/dashboard",
    label: "Dashboard",
    icon: LayoutDashboard,
  },
  {
    href: "/dashboard/assets",
    label: "Assets",
    icon: Package,
  },
  {
    href: "/dashboard/anomalies",
    label: "Anomalies",
    icon: AlertTriangle,
  },
  {
    href: "/dashboard/rul",
    label: "RUL Predictions",
    icon: TrendingUp,
  },
  {
    href: "/dashboard/interventions",
    label: "Interventions",
    icon: Wrench,
  },
  {
    href: "/dashboard/map",
    label: "Map",
    icon: Map,
  },
  {
    href: "/dashboard/reports",
    label: "Reports",
    icon: FileText,
  },
]

export function Sidebar() {
  const pathname = usePathname()
  const router = useRouter()
  const { logout, user } = useAuth()
  const [isOpen, setIsOpen] = useState(false)

  const handleLogout = () => {
    logout()
    router.push("/login")
  }

  return (
    <>
      {/* Mobile menu button */}
      <div className="lg:hidden fixed top-4 left-4 z-40">
        <Button variant="outline" size="icon" onClick={() => setIsOpen(!isOpen)} className="h-10 w-10">
          {isOpen ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
        </Button>
      </div>

      {/* Sidebar */}
      <aside
        className={cn(
          "fixed left-0 top-0 z-50 h-screen w-64 bg-sidebar border-r border-sidebar-border transition-transform duration-300 ease-in-out lg:translate-x-0",
          isOpen ? "translate-x-0" : "-translate-x-full",
        )}
      >
        <div className="flex flex-col h-full">
          {/* Logo */}
          <div className="px-6 py-8">
            <Link href="/dashboard" className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-lg bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center">
                <svg className="h-6 w-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                  />
                </svg>
              </div>
              <div>
                <h1 className="font-bold text-lg text-sidebar-foreground">PredMaint</h1>
                <p className="text-xs text-sidebar-accent-foreground">Dashboard</p>
              </div>
            </Link>
          </div>

          {/* Navigation Links */}
          <nav className="flex-1 space-y-1 px-4">
            {sidebarLinks.map(({ href, label, icon: Icon }) => {
              const isActive = pathname === href || pathname.startsWith(href + "/")
              return (
                <Link
                  key={href}
                  href={href}
                  onClick={() => setIsOpen(false)}
                  className={cn(
                    "flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-colors",
                    isActive
                      ? "bg-sidebar-primary text-sidebar-primary-foreground"
                      : "text-sidebar-foreground hover:bg-sidebar-accent",
                  )}
                >
                  <Icon className="h-4 w-4" />
                  {label}
                </Link>
              )
            })}
          </nav>

          {/* User section */}
          <div className="border-t border-sidebar-border p-4 space-y-4">
            <div className="px-2 py-3">
              <p className="text-xs text-sidebar-accent-foreground uppercase tracking-wide">Logged in as</p>
              <p className="font-medium text-sidebar-foreground truncate">{user?.email}</p>
            </div>
            <Button onClick={handleLogout} variant="outline" className="w-full justify-start gap-2 bg-transparent">
              <LogOut className="h-4 w-4" />
              Logout
            </Button>
          </div>
        </div>
      </aside>

      {/* Overlay for mobile */}
      {isOpen && <div className="fixed inset-0 z-40 bg-black/50 lg:hidden" onClick={() => setIsOpen(false)} />}
    </>
  )
}
