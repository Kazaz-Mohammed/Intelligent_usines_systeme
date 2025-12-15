"use client"

import { useEffect } from "react"
import { useRouter } from "next/navigation"
import { useAuth } from "@/context/auth-context"

export default function HomePage() {
  const router = useRouter()
  const { token, isLoading } = useAuth()

  useEffect(() => {
    if (!isLoading) {
      if (token) {
        router.push("/dashboard")
      } else {
        router.push("/login")
      }
    }
  }, [token, isLoading, router])

  return (
    <div className="flex items-center justify-center h-screen">
      <div className="text-center">
        <div className="h-12 w-12 rounded-full border-4 border-border border-t-primary animate-spin mx-auto mb-4" />
        <p className="text-muted-foreground">Redirecting...</p>
      </div>
    </div>
  )
}
