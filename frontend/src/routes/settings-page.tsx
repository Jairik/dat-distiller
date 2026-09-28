/**
 * Settings: Providers, API keys, local CLI detection — status view.
 *
 * Keys never display in full: the API only exposes {set, source}. Editing
 * keys and limits lands with the settings feature issue.
 */

import { useQuery } from '@tanstack/react-query'
import { apiGet } from '@/api/client'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

interface KeyStatus {
  set: boolean
  source: 'env' | 'stored' | 'none' | null
}

interface ProviderInfo {
  id: string
  kind: 'cli' | 'http'
  available: boolean
}

interface SettingsResponse extends Record<string, unknown> {
  keys: Record<string, KeyStatus>
  providers: ProviderInfo[]
}

export function SettingsPage() {
  const settings = useQuery({
    queryKey: ['settings'],
    queryFn: () => apiGet<SettingsResponse>('/settings'),
  })

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Settings</h1>
        <p className="text-sm text-muted-foreground">
          Everything stays on this machine; keys are stored with 0600 permissions.
        </p>
      </div>
      {settings.isPending && <p className="text-sm text-muted-foreground">Loading…</p>}
      {settings.isError && (
        <p role="alert" className="text-sm text-destructive">
          Could not load settings: {(settings.error as Error).message}
        </p>
      )}
      {settings.data && (
        <>
          <Card>
            <CardHeader>
              <CardTitle>API keys</CardTitle>
              <CardDescription>Environment variables win over stored keys.</CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-3">
              {Object.entries(settings.data.keys).map(([provider, status]) => (
                <div key={provider} className="flex items-center justify-between gap-3">
                  <span className="text-sm capitalize">{provider}</span>
                  <Badge variant={status.set ? 'default' : 'outline'} className="capitalize">
                    {status.set ? `set (${status.source})` : 'not set'}
                  </Badge>
                </div>
              ))}
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Providers</CardTitle>
              <CardDescription>Local CLIs are detected on PATH at startup.</CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-3">
              {settings.data.providers.map((provider) => (
                <div key={provider.id} className="flex items-center justify-between gap-3">
                  <span className="text-sm">
                    {provider.id}
                    <span className="ml-2 text-xs text-muted-foreground">{provider.kind}</span>
                  </span>
                  <Badge variant={provider.available ? 'default' : 'outline'}>
                    {provider.available ? 'available' : 'unavailable'}
                  </Badge>
                </div>
              ))}
            </CardContent>
          </Card>
        </>
      )}
    </div>
  )
}
