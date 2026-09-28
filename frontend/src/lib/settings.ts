/**
 * Settings: preferences, API key status, Provider availability, extras.
 *
 * Two rules shape everything here:
 *
 * - Key values never cross the wire. The API reports `{set, source}` only, so
 *   every key field is write-only: what you type is sent once and dropped.
 * - An env key wins over a stored one (backend `settings.py`), so it is shown
 *   as read-only rather than as something this page can overwrite.
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiDelete, apiGet, apiPut } from '@/api/client'

// ---------------------------------------------------------------------------
// Types (mirrors `api/settings.py` and `api/health.py`)
// ---------------------------------------------------------------------------

export interface KeyStatus {
  set: boolean
  /** Where the effective key came from: the environment, or keys.json. */
  source: 'env' | 'file' | null
}

export interface ProviderInfo {
  id: string
  kind: 'cli' | 'http'
  available: boolean
}

export interface SoftLimits {
  upload_rows: number
  generation_rows: number
  labeling_calls: number
}

export interface SettingsData {
  default_provider: string
  /** provider id -> model id; `""` means "let the Provider decide". */
  models: Record<string, string>
  soft_limits: SoftLimits
  review_threshold: number
  fairness_gap_threshold: number
  keys: Record<string, KeyStatus>
  providers: ProviderInfo[]
}

/** Body of `PUT /api/settings`; the backend merges and validates each key. */
export interface SettingsPatch {
  default_provider?: string
  models?: Record<string, string>
  soft_limits?: Partial<SoftLimits>
  review_threshold?: number
  fairness_gap_threshold?: number
}

export interface HealthData {
  status: string
  version: string
  extras: Record<string, boolean>
}

export interface OpenRouterModel {
  id: string
  label: string
}

/** Model id meaning "use whatever this Provider calls by default". */
export const AGENT_DEFAULT = ''

/** What a set key is displayed as — the real value is never fetched. */
export const KEY_MASK = '••••••••••••'

// ---------------------------------------------------------------------------
// Copy + install metadata
// ---------------------------------------------------------------------------

export const PROVIDER_NAMES: Record<string, string> = {
  claude: 'Claude Code',
  codex: 'Codex',
  opencode: 'OpenCode',
  openrouter: 'OpenRouter',
  typesafe: 'Jev (TypeSafe)',
}

export function providerName(id: string): string {
  return PROVIDER_NAMES[id] ?? id
}

/** Install command for a local CLI Provider that is not on PATH. */
export const CLI_INSTALL_COMMANDS: Record<string, string> = {
  claude: 'npm i -g @anthropic-ai/claude-code',
  codex: 'npm i -g @openai/codex',
  opencode: 'npm i -g opencode-ai',
}

export const EXTRA_NAMES: Record<string, string> = {
  torch: 'Torch',
  tensorflow: 'TensorFlow',
  presidio: 'Presidio',
}

export const EXTRA_PURPOSES: Record<string, string> = {
  torch: 'Neural net models',
  tensorflow: 'TensorFlow models',
  presidio: 'PII detection with named entities',
}

export function extraName(extra: string): string {
  return EXTRA_NAMES[extra] ?? extra
}

export function extraInstallCommand(extra: string): string {
  return `uv sync --extra ${extra}`
}

/** Env var the backend prefers over a stored key (mirrors KEY_ENV_VARS). */
export const KEY_ENV_VARS: Record<string, string> = {
  openrouter: 'OPENROUTER_API_KEY',
  typesafe: 'TYPESAFE_API_KEY',
}

export const KEY_PURPOSES: Record<string, string> = {
  openrouter: ' lets Providers generate rows through OpenRouter.',
  typesafe: ' lets Jev answer Jev Questions.',
}

export interface SoftLimitField {
  key: keyof SoftLimits
  label: string
  hint: string
}

/** Labels follow CONTEXT.md: Generation, Labeling, Dataset Version, Check. */
export const SOFT_LIMIT_FIELDS: SoftLimitField[] = [
  {
    key: 'upload_rows',
    label: 'Upload rows',
    hint: 'Rows in an uploaded Dataset Version before a Check warns.',
  },
  {
    key: 'generation_rows',
    label: 'Generation rows',
    hint: 'Rows one Generation produces before a Check warns.',
  },
  {
    key: 'labeling_calls',
    label: 'Labeling calls',
    hint: 'Jev calls one Labeling run makes before a Check warns.',
  },
]

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

export function useSettings() {
  return useQuery({
    queryKey: ['settings'],
    queryFn: () => apiGet<SettingsData>('/settings'),
  })
}

export function useHealth() {
  return useQuery({
    queryKey: ['health'],
    queryFn: () => apiGet<HealthData>('/health'),
    retry: false,
  })
}

/**
 * OpenRouter's catalogue for the model picker. Only worth asking when a key
 * exists: without one the backend answers 422 and the UI falls back to a
 * free-text model id.
 */
export function useOpenRouterModels(enabled: boolean) {
  return useQuery({
    queryKey: ['openrouter-models'],
    queryFn: async () => {
      const body = await apiGet<{ models: Array<{ id: string; name?: string; label?: string }> }>(
        '/providers/openrouter/models',
      )
      return body.models.map((model) => ({
        id: model.id,
        label: model.label ?? model.name ?? model.id,
      })) satisfies OpenRouterModel[]
    },
    enabled,
    retry: false,
    staleTime: 5 * 60 * 1000,
  })
}

// ---------------------------------------------------------------------------
// Mutations
// ---------------------------------------------------------------------------

/** Shared invalidation so every section reflects a saved change. */
function useInvalidateSettings() {
  const queryClient = useQueryClient()
  return () => {
    queryClient.invalidateQueries({ queryKey: ['settings'] })
    queryClient.invalidateQueries({ queryKey: ['openrouter-models'] })
  }
}

export function useSaveSettings() {
  const invalidate = useInvalidateSettings()
  return useMutation({
    mutationFn: (patch: SettingsPatch) => apiPut<SettingsData>('/settings', patch),
    onSuccess: invalidate,
  })
}

export function useSetKey() {
  const invalidate = useInvalidateSettings()
  return useMutation({
    mutationFn: ({ provider, key }: { provider: string; key: string }) =>
      apiPut<{ keys: Record<string, KeyStatus> }>(`/settings/keys/${provider}`, { key }),
    onSuccess: invalidate,
  })
}

export function useClearKey() {
  const invalidate = useInvalidateSettings()
  return useMutation({
    mutationFn: (provider: string) => apiDelete(`/settings/keys/${provider}`),
    onSuccess: invalidate,
  })
}
