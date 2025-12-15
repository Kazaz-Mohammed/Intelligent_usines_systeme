"use client"

import { useAssets } from "@/hooks/useAssets"
import { AssetsTable } from "@/components/tables/assets-table"

export default function AssetsPage() {
  const { assets, isLoading, error } = useAssets({ page_size: 100 })

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Assets</h1>
        <p className="text-muted-foreground mt-2">Manage and monitor your industrial assets</p>
      </div>

      {error && (
        <div className="bg-destructive/10 text-destructive p-4 rounded-lg">
          {error}
        </div>
      )}

      <AssetsTable assets={assets} isLoading={isLoading} />
    </div>
  )
}
