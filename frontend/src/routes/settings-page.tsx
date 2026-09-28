/**
 * Settings: API keys, Providers (availability + model id + install hints),
 * soft limits and thresholds, optional extras.
 *
 * Keys are write-only by design: the API reports `{set, source}` and never a
 * value, so a typed key is sent once and dropped from state. A key that comes
 * from an environment variable is read-only here because the backend always
 * prefers the environment over the stored file.
 */

import { useEffect, useState } from 'react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { CopyButton } from '@/components/settings/copy-button'
import { SavedPill, useSavedFlash, type SavedNotifier } from '@/components/settings/saved-pill'
import {
  AGENT_DEFAULT,
  CLI_INSTALL_COMMANDS,
  EXTRA_PURPOSES,
  KEY_ENV_VARS,
  KEY_MASK,
  KEY_PURPOSES,
  SOFT_LIMIT_FIELDS,
  extraInstallCommand,
  extraName,
  providerName,
  useClearKey,
  useHealth,
  useOpenRouterModels,
  useSaveSettings,
  useSetKey,
  useSettings,
  type KeyStatus,
  type ProviderInfo,
  type SettingsData,
} from '@/lib/settings'

/** Native select keeps mobile pickers and needs no portal; styled to match. */
const SELECT_CLASS =
  'h-9 w-full min-w-0 rounded-md border border-input bg-transparent px-3 py-1 text-base shadow-xs outline-none transition-[color,box-shadow] focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 disabled:pointer-events-none disabled:opacity-50 md:text-sm dark:bg-input/30'

const CUSTOM_MODEL = '__custom__'

export function SettingsPage() {
  const settings = useSettings()
  const { flash, saved } = useSavedFlash()

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
          <KeysSection keys={settings.data.keys} onSaved={saved} />
          <ProvidersSection settings={settings.data} onSaved={saved} />
          <LimitsSection settings={settings.data} onSaved={saved} />
          <ExtrasSection />
        </>
      )}

      <SavedPill flash={flash} />
    </div>
  )
}

// ---------------------------------------------------------------------------
// API keys
// ---------------------------------------------------------------------------

function KeysSection({
  keys,
  onSaved,
}: {
  keys: Record<string, KeyStatus>
  onSaved: SavedNotifier
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>API keys</CardTitle>
        <CardDescription>
          Values are never sent back to the browser, so this is a write-only field: saving a new
          key replaces the stored one. Environment variables win over stored keys.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        {Object.entries(keys).map(([provider, status]) => (
          <KeyRow key={provider} provider={provider} status={status} onSaved={onSaved} />
        ))}
      </CardContent>
    </Card>
  )
}

function KeyRow({
  provider,
  status,
  onSaved,
}: {
  provider: string
  status: KeyStatus
  onSaved: SavedNotifier
}) {
  const setKey = useSetKey()
  const clearKey = useClearKey()
  const [value, setValue] = useState('')
  const fromEnv = status.source === 'env'
  const envVar = KEY_ENV_VARS[provider] ?? 'an environment variable'
  const name = providerName(provider)
  const error = setKey.error ?? clearKey.error

  async function submit() {
    const key = value.trim()
    if (!key) return
    try {
      await setKey.mutateAsync({ provider, key })
      setValue('') // write-only: never echo the value back into the field
      onSaved(`${name} key saved`)
    } catch {
      // surfaced inline below
    }
  }

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-border p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          <span className="text-sm font-medium">{name}</span>
          <Badge variant={status.set ? 'default' : 'outline'}>
            {status.set ? `set (${status.source})` : 'not set'}
          </Badge>
          {status.set && (
            <span
              className="font-mono text-xs text-muted-foreground"
              title="The value is never sent to the browser"
            >
              {KEY_MASK}
            </span>
          )}
        </div>
        {status.set && !fromEnv && (
          <Button
            size="xs"
            variant="ghost"
            disabled={clearKey.isPending}
            onClick={async () => {
              await clearKey.mutateAsync(provider)
              onSaved(`${name} key cleared`)
            }}
          >
            Clear
          </Button>
        )}
      </div>

      <p className="text-xs text-muted-foreground">
        <span className="capitalize">{provider}</span>
        {KEY_PURPOSES[provider] ?? ' key.'}
      </p>

      {fromEnv ? (
        <p className="rounded-md bg-muted/50 px-2.5 py-2 text-xs text-muted-foreground">
          Read-only: this key comes from the <code className="font-mono">{envVar}</code>{' '}
          environment variable, which the backend always prefers over a stored key. Editing it here
          would have no effect — change the environment instead and reload this page.
        </p>
      ) : (
        <div className="flex flex-wrap items-end gap-2">
          <div className="flex min-w-52 flex-1 flex-col gap-1.5">
            <Label htmlFor={`key-${provider}`}>New {name} key</Label>
            <Input
              id={`key-${provider}`}
              type="password"
              autoComplete="off"
              spellCheck={false}
              placeholder={status.set ? 'Replace stored key' : 'Paste a key'}
              value={value}
              onChange={(event) => setValue(event.target.value)}
              onKeyDown={(event) => event.key === 'Enter' && void submit()}
            />
          </div>
          <Button size="sm" disabled={!value.trim() || setKey.isPending} onClick={submit}>
            {setKey.isPending ? 'Saving…' : 'Save key'}
          </Button>
        </div>
      )}

      {error && (
        <p role="alert" className="text-xs text-destructive">
          {(error as Error).message}
        </p>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Providers
// ---------------------------------------------------------------------------

function ProvidersSection({
  settings,
  onSaved,
}: {
  settings: SettingsData
  onSaved: SavedNotifier
}) {
  const save = useSaveSettings()
  const [defaultProvider, setDefaultProvider] = useState(settings.default_provider)
  useEffect(() => setDefaultProvider(settings.default_provider), [settings.default_provider])
  const error = save.error

  return (
    <Card>
      <CardHeader>
        <CardTitle>Providers</CardTitle>
        <CardDescription>
          Local CLI Providers are detected on PATH; OpenRouter is ready once its key is set.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div className="flex flex-wrap items-end gap-2 rounded-lg border border-border p-3">
          <div className="flex min-w-52 flex-1 flex-col gap-1.5">
            <Label htmlFor="default-provider">Default Provider</Label>
            <select
              id="default-provider"
              className={SELECT_CLASS}
              value={defaultProvider}
              onChange={async (event) => {
                setDefaultProvider(event.target.value)
                await save.mutateAsync({ default_provider: event.target.value })
                onSaved('Default Provider saved')
              }}
            >
              {settings.providers.map((provider) => (
                <option key={provider.id} value={provider.id}>
                  {providerName(provider.id)}
                  {provider.available ? '' : ' (unavailable)'}
                </option>
              ))}
            </select>
          </div>
        </div>

        {settings.providers.map((provider) => (
          <ProviderRow
            key={provider.id}
            provider={provider}
            settings={settings}
            onSaved={onSaved}
          />
        ))}

        {error && (
          <p role="alert" className="text-xs text-destructive">
            {(error as Error).message}
          </p>
        )}
      </CardContent>
    </Card>
  )
}

function availabilityLabel(provider: ProviderInfo): string {
  if (provider.available) return 'available'
  return provider.kind === 'http' ? 'no key set' : 'not installed'
}

function ProviderRow({
  provider,
  settings,
  onSaved,
}: {
  provider: ProviderInfo
  settings: SettingsData
  onSaved: SavedNotifier
}) {
  const name = providerName(provider.id)
  const install = CLI_INSTALL_COMMANDS[provider.id]
  const keyStatus = settings.keys[provider.id]

  return (
    <div className="flex flex-col gap-3 rounded-lg border border-border p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm font-medium">{name}</span>
          <Badge variant="outline" className="uppercase">
            {provider.kind}
          </Badge>
        </div>
        <Badge variant={provider.available ? 'default' : 'outline'}>
          {availabilityLabel(provider)}
        </Badge>
      </div>

      {!provider.available && install && (
        <div className="flex flex-col gap-1.5">
          <p className="text-xs text-muted-foreground">
            Not on PATH. Install it, then reload this page:
          </p>
          <CopyButton text={install} label={`${name} install command`} />
        </div>
      )}

      {!provider.available && provider.kind === 'http' && (
        <p className="text-xs text-muted-foreground">
          Set the {name} key above to use this Provider.
        </p>
      )}

      <ModelField
        provider={provider.id}
        kind={provider.kind}
        value={settings.models[provider.id] ?? AGENT_DEFAULT}
        keySet={Boolean(keyStatus?.set)}
        onSaved={onSaved}
      />
    </div>
  )
}

function ModelField({
  provider,
  kind,
  value,
  keySet,
  onSaved,
}: {
  provider: string
  kind: 'cli' | 'http'
  value: string
  keySet: boolean
  onSaved: SavedNotifier
}) {
  const save = useSaveSettings()
  // Only OpenRouter has a catalogue. Every Provider row shares one query key,
  // so the data is gated on `kind` here — otherwise a CLI Provider would offer
  // OpenRouter model ids it cannot call.
  const wantsCatalogue = kind === 'http' && keySet
  const models = useOpenRouterModels(wantsCatalogue)
  // No catalogue to offer (a CLI Provider, or OpenRouter without a usable
  // listing) → the free-text input is what the user needs to see right away.
  const noCatalogue = !wantsCatalogue || models.isError
  const catalogue = noCatalogue ? [] : (models.data ?? [])
  const [customOpen, setCustomOpen] = useState(noCatalogue)
  const [draft, setDraft] = useState(value)
  useEffect(() => setCustomOpen(noCatalogue), [noCatalogue])
  useEffect(() => setDraft(value), [value])

  const name = providerName(provider)
  const known = catalogue.some((model) => model.id === value)

  async function commit(next: string, message = `${name} model saved`) {
    try {
      await save.mutateAsync({ models: { [provider]: next } })
      onSaved(message)
      return true
    } catch {
      return false
    }
  }

  return (
    <div className="flex flex-col gap-1.5">
      <Label htmlFor={`model-${provider}`}>Model</Label>
      <select
        id={`model-${provider}`}
        className={SELECT_CLASS}
        value={customOpen ? CUSTOM_MODEL : value}
        onChange={(event) => {
          const next = event.target.value
          if (next === CUSTOM_MODEL) {
            setCustomOpen(true)
            return
          }
          setCustomOpen(false)
          void commit(next === AGENT_DEFAULT ? next : next, next === AGENT_DEFAULT ? `${name} uses agent default` : undefined)
        }}
      >
        <option value={AGENT_DEFAULT}>agent default</option>
        {catalogue.map((model) => (
          <option key={model.id} value={model.id}>
            {model.label}
          </option>
        ))}
        {value && !known && <option value={value}>{value} (stored)</option>}
        <option value={CUSTOM_MODEL}>another model id…</option>
      </select>

      {customOpen ? (
        <div className="flex flex-wrap items-end gap-2">
          <div className="flex min-w-52 flex-1 flex-col gap-1.5">
            <Label htmlFor={`model-id-${provider}`}>Model id</Label>
            <Input
              id={`model-id-${provider}`}
              placeholder={kind === 'http' ? 'anthropic/claude-sonnet-4.5' : 'gpt-5-codex'}
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
            />
          </div>
          <Button
            size="sm"
            disabled={!draft.trim() || save.isPending}
            onClick={async () => {
              if (await commit(draft.trim())) setCustomOpen(false)
            }}
          >
            {save.isPending ? 'Saving…' : 'Save model'}
          </Button>
          {value && (
            <Button
              size="sm"
              variant="ghost"
              disabled={save.isPending}
              onClick={async () => {
                if (await commit(AGENT_DEFAULT, `${name} uses agent default`)) setCustomOpen(false)
              }}
            >
              Use agent default
            </Button>
          )}
        </div>
      ) : (
        <p className="text-xs text-muted-foreground">
          {catalogue.length > 0
            ? 'From the OpenRouter model list.'
            : 'Free-text: type the model id this Provider should call.'}
        </p>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Soft limits + thresholds
// ---------------------------------------------------------------------------

interface LimitsDraft {
  upload_rows: string
  generation_rows: string
  labeling_calls: string
  review_threshold: string
  fairness_gap_threshold: string
}

function toDraft(settings: SettingsData): LimitsDraft {
  return {
    upload_rows: String(settings.soft_limits.upload_rows),
    generation_rows: String(settings.soft_limits.generation_rows),
    labeling_calls: String(settings.soft_limits.labeling_calls),
    review_threshold: String(settings.review_threshold),
    fairness_gap_threshold: String(settings.fairness_gap_threshold),
  }
}

function LimitsSection({ settings, onSaved }: { settings: SettingsData; onSaved: SavedNotifier }) {
  const save = useSaveSettings()
  const [draft, setDraft] = useState(() => toDraft(settings))
  // Only re-sync when the server values actually change, so typing is never
  // wiped by an unrelated refetch.
  const serverKey = JSON.stringify(toDraft(settings))
  useEffect(() => setDraft(toDraft(settings)), [serverKey])
  const [invalid, setInvalid] = useState<string | null>(null)

  const patch = (key: keyof LimitsDraft) => (event: { target: { value: string } }) => {
    setInvalid(null)
    setDraft((current) => ({ ...current, [key]: event.target.value }))
  }

  async function submit() {
    const limits: Record<string, number> = {}
    for (const field of SOFT_LIMIT_FIELDS) {
      const parsed = Number(draft[field.key])
      if (!Number.isFinite(parsed) || parsed < 1) {
        setInvalid(`${field.label} must be a whole number of at least 1.`)
        return
      }
      limits[field.key] = Math.trunc(parsed)
    }
    const review = Number(draft.review_threshold)
    const fairness = Number(draft.fairness_gap_threshold)
    if (!(review >= 0 && review <= 1) || !(fairness >= 0 && fairness <= 1)) {
      setInvalid('Thresholds are proportions between 0 and 1.')
      return
    }
    try {
      await save.mutateAsync({
        soft_limits: limits,
        review_threshold: review,
        fairness_gap_threshold: fairness,
      })
      onSaved('Limits saved')
    } catch {
      // surfaced inline below
    }
  }

  const error = invalid ?? (save.isError ? (save.error as Error).message : null)

  return (
    <Card>
      <CardHeader>
        <CardTitle>Soft limits and thresholds</CardTitle>
        <CardDescription>
          A soft limit never blocks a step; crossing it raises a Check warning. Jev labels below the
          review threshold go to the Review Queue.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <div className="grid gap-4 sm:grid-cols-3">
          {SOFT_LIMIT_FIELDS.map((field) => (
            <div key={field.key} className="flex flex-col gap-1.5">
              <Label htmlFor={`limit-${field.key}`}>{field.label}</Label>
              <Input
                id={`limit-${field.key}`}
                type="number"
                min={1}
                step={1}
                inputMode="numeric"
                value={draft[field.key]}
                onChange={patch(field.key)}
              />
              <p className="text-xs text-muted-foreground">{field.hint}</p>
            </div>
          ))}
        </div>
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="review-threshold">Review threshold</Label>
            <Input
              id="review-threshold"
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={draft.review_threshold}
              onChange={patch('review_threshold')}
            />
            <p className="text-xs text-muted-foreground">
              Confidence under this lands a Label Column in the Review Queue.
            </p>
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="fairness-threshold">Fairness gap threshold</Label>
            <Input
              id="fairness-threshold"
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={draft.fairness_gap_threshold}
              onChange={patch('fairness_gap_threshold')}
            />
            <p className="text-xs text-muted-foreground">
              Per-group metric gap in a Fairness Report that raises a Check.
            </p>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <Button size="sm" disabled={save.isPending} onClick={submit}>
            {save.isPending ? 'Saving…' : 'Save limits'}
          </Button>
          {error && (
            <p role="alert" className="text-xs text-destructive">
              {error}
            </p>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Optional extras
// ---------------------------------------------------------------------------

function ExtrasSection() {
  const health = useHealth()
  const extras = Object.entries(health.data?.extras ?? {})

  return (
    <Card>
      <CardHeader>
        <CardTitle>Optional extras</CardTitle>
        <CardDescription>
          Heavy features degrade gracefully when these are not installed.
          {health.data ? ` Running dat-distiller ${health.data.version}.` : ''}
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        {health.isPending && <p className="text-sm text-muted-foreground">Checking…</p>}
        {health.isError && (
          <p className="text-sm text-muted-foreground">
            Could not read install status from /api/health.
          </p>
        )}
        {extras.map(([extra, installed]) => {
          const name = extraName(extra)
          return (
            <div key={extra} className="flex flex-col gap-2 rounded-lg border border-border p-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-sm font-medium">{name}</span>
                  <span className="font-mono text-xs text-muted-foreground">{extra}</span>
                </div>
                <Badge variant={installed ? 'default' : 'outline'}>
                  {installed ? 'installed' : 'missing'}
                </Badge>
              </div>
              <p className="text-xs text-muted-foreground">
                {EXTRA_PURPOSES[extra] ?? 'Optional capability.'}
              </p>
              {!installed && (
                <CopyButton text={extraInstallCommand(extra)} label={`${name} install command`} />
              )}
            </div>
          )
        })}
      </CardContent>
    </Card>
  )
}
