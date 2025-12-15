"use client"

import { useEffect, useRef, useCallback, useState } from "react"
import type { WebSocketMessage } from "@/types"

interface WebSocketHookOptions {
  url?: string
  onMessage?: (message: WebSocketMessage) => void
  onConnect?: () => void
  onDisconnect?: () => void
  onError?: (error: Event) => void
}

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8091/ws/dashboard"
const PING_INTERVAL = 25000 // Send ping every 25 seconds
const RECONNECT_DELAY = 3000 // 3 seconds

// Singleton WebSocket connection to survive React Strict Mode
let globalWs: WebSocket | null = null
let globalWsUrl: string | null = null
let connectionCount = 0

export function useWebSocket(options: WebSocketHookOptions = {}) {
  const {
    url = WS_URL,
    onMessage,
    onConnect,
    onDisconnect,
    onError,
  } = options

  const [isConnected, setIsConnected] = useState(false)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>()
  const pingIntervalRef = useRef<NodeJS.Timeout>()
  const mountedRef = useRef(true)
  const optionsRef = useRef(options)

  // Keep options ref up to date
  useEffect(() => {
    optionsRef.current = options
  }, [options])

  const connect = useCallback(() => {
    if (typeof window === "undefined") return

    // If we have an existing connection to the same URL and it's open/connecting, reuse it
    if (globalWs && globalWsUrl === url && 
        (globalWs.readyState === WebSocket.OPEN || globalWs.readyState === WebSocket.CONNECTING)) {
      console.log("Reusing existing WebSocket connection")
      if (globalWs.readyState === WebSocket.OPEN) {
        setIsConnected(true)
      }
      return
    }

    // Close any existing connection to different URL
    if (globalWs && globalWsUrl !== url) {
      globalWs.close()
      globalWs = null
      globalWsUrl = null
    }

    console.log(`Creating new WebSocket connection to: ${url}`)
    
    try {
      const ws = new WebSocket(url)
      globalWs = ws
      globalWsUrl = url

      ws.onopen = () => {
        console.log("WebSocket connected successfully!")
        if (mountedRef.current) {
          setIsConnected(true)
          optionsRef.current.onConnect?.()
        }

        // Start sending pings to keep connection alive
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current)
        }
        pingIntervalRef.current = setInterval(() => {
          if (globalWs && globalWs.readyState === WebSocket.OPEN) {
            try {
              globalWs.send(JSON.stringify({ type: "ping" }))
            } catch (err) {
              console.error("Failed to send ping:", err)
            }
          }
        }, PING_INTERVAL)
      }

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data)
          // Handle pong and ping messages silently
          if (message.type === "pong" || message.type === "ping") {
            return
          }
          if (mountedRef.current) {
            optionsRef.current.onMessage?.(message)
          }
        } catch (err) {
          console.error("Failed to parse WebSocket message:", err)
        }
      }

      ws.onerror = (error) => {
        console.error("WebSocket error:", error)
        if (mountedRef.current) {
          optionsRef.current.onError?.(error)
        }
      }

      ws.onclose = (event) => {
        console.log(`WebSocket disconnected (code: ${event.code}, reason: ${event.reason || 'none'})`)
        globalWs = null
        globalWsUrl = null
        
        if (mountedRef.current) {
          setIsConnected(false)
          optionsRef.current.onDisconnect?.()
        }

        // Clear ping interval
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current)
          pingIntervalRef.current = undefined
        }

        // Auto-reconnect if component is still mounted
        if (mountedRef.current && connectionCount > 0) {
          if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current)
          }
          reconnectTimeoutRef.current = setTimeout(() => {
            console.log("Attempting to reconnect WebSocket...")
            if (mountedRef.current) {
              connect()
            }
          }, RECONNECT_DELAY)
        }
      }
    } catch (err) {
      console.error("Failed to create WebSocket:", err)
    }
  }, [url])

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = undefined
    }
    
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current)
      pingIntervalRef.current = undefined
    }
    
    // Only close if no other hooks are using it
    connectionCount--
    if (connectionCount <= 0 && globalWs) {
      console.log("Closing WebSocket connection (no more subscribers)")
      globalWs.close()
      globalWs = null
      globalWsUrl = null
      connectionCount = 0
    }
  }, [])

  const send = useCallback((message: WebSocketMessage) => {
    if (globalWs && globalWs.readyState === WebSocket.OPEN) {
      globalWs.send(JSON.stringify(message))
    }
  }, [])

  useEffect(() => {
    mountedRef.current = true
    connectionCount++
    
    // Small delay to handle React Strict Mode double-mounting
    const connectTimeout = setTimeout(() => {
      if (mountedRef.current) {
        connect()
      }
    }, 100)

    return () => {
      mountedRef.current = false
      clearTimeout(connectTimeout)
      disconnect()
    }
  }, [connect, disconnect])

  // Sync isConnected state with global WebSocket
  useEffect(() => {
    const checkConnection = () => {
      const connected = globalWs?.readyState === WebSocket.OPEN
      setIsConnected(connected)
    }
    
    // Check immediately
    checkConnection()
    
    // Check periodically
    const interval = setInterval(checkConnection, 1000)
    return () => clearInterval(interval)
  }, [])

  return {
    isConnected,
    send,
    disconnect: () => {
      connectionCount = 0 // Force close
      if (globalWs) {
        globalWs.close()
        globalWs = null
        globalWsUrl = null
      }
    },
  }
}
