/**
 * The Generate step's data layer: Column Specs, the sample/specs choice,
 * preview, estimate and run.
 *
 * The order the endpoints are called in is the contract the UI enforces: a
 * **preview must succeed before a run can start**. `useGenerationStep` is the
 * only place that order lives, so the buttons cannot get it wrong.
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiGet, apiPost } from '@/api/client'

// -- types (mirror `api/generate.py` and `profile.py`) ------------------------

/** `SPEC_TYPES` on the backend, in the order the picker shows them. */
export const SPEC_TYPES = [
  'number',
  'integer',
  'categorical',
  'bool',
  'datetime',
  'text',
] as const
export type SpecType = (typeof SPEC_TYPES)[number]

export interface ColumnSpec {
  name: string
  type: SpecType
  min: number | null
  max: number | null
  categories: string[] | null
  description: string | null
}

export const MODES = ['hybrid', 'statistical', 'llm'] as const
export type GenerationMode = (typeof MODES)[number]

export interface ProviderOption {
  id: string
  available: boolean
  reason: string | null
  model: string | null
}

export interface PreviewResult {
  seed: number
  rows: Array<Record<string, unknown>>
  provider_calls: number | null
}

export interface EstimateResult {
  rows: number
  estimated_provider_calls: number
  uses_provider: boolean
  seed?: number
  warnings?: string[]
  notes?: string[]
}

export interface GenerateRunResult {
  version_id: string
  rows: number
  seed: number
  provider_calls: number
  dropped_rows: number
  fidelity_warnings: string[]
  checks_raised: string[]
  pii?: { columns: string[]; total_findings: number }
}

/** The body every Generate endpoint takes. */
export interface GenerationBody {
  project_id: string
  description: string
  mode: GenerationMode
  count: number
  specs?: ColumnSpec[]
  sample_version_id?: string
  balance?: Record<string, Record<string, number>>
  provider?: string
  model?: string
  seed?: number
}

// -- the Column Specs table ---------------------------------------------------

export function emptySpec(index: number): ColumnSpec {
  return {
    name: `column_${index + 1}`,
    type: 'number',
    min: null,
    max: null,
    categories: null,
    description: null,
  }
}

/**
 * Normalize one editable row into a `ColumnSpec`. Accepts a form row (strings,
 * blanks) or an existing spec — the table round-trips both through here, so
 * there is a single place where "blank means null" is decided.
 */
export function toSpec(
  row: ColumnSpec | Record<string, unknown>,
  _index = 0,
): ColumnSpec {
  const num = (value: unknown): number | null => {
    if (value === '' || value === null || value === undefined) return null
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : null
  }
  const text = (value: unknown): string | null => {
    const trimmed = String(value ?? '').trim()
    return trimmed || null
  }
  const type = SPEC_TYPES.includes(row.type as SpecType) ? (row.type as SpecType) : 'number'
  return {
    // An emptied name stays empty so the table can report it as invalid,
    // rather than silently reverting to a generated default.
    name: text(row.name) ?? '',
    type,
    min: num(row.min),
    max: num(row.max),
    categories:
      type === 'categorical'
        ? (Array.isArray(row.categories) ? row.categories.join(',') : String(row.categories ?? ''))
            .split(',')
            .map((part) => part.trim())
            .filter(Boolean)
        : null,
    description: text(row.description),
  }
}

/** Trim to what the API accepts, so a half-typed row is not sent as garbage. */
export function toWireSpec(spec: ColumnSpec): Record<string, unknown> {
  const out: Record<string, unknown> = { name: spec.name, type: spec.type }
  if (spec.min !== null) out.min = spec.min
  if (spec.max !== null) out.max = spec.max
  if (spec.categories?.length) out.categories = spec.categories
  if (spec.description) out.description = spec.description
  return out
}

/** The API's own validation, so the UI never re-implements the rules. */
export function localSpecErrors(specs: ColumnSpec[]): string[] {
  const errors: string[] = []
  const seen = new Set<string>()
  specs.forEach((spec, index) => {
    const where = spec.name.trim() || `column ${index + 1}`
    if (!spec.name.trim()) errors.push(`${where}: name is required`)
    if (seen.has(spec.name.trim())) errors.push(`${where}: duplicate name`)
    seen.add(spec.name.trim())
    if (spec.min !== null && spec.max !== null && spec.min > spec.max) {
      errors.push(`${where}: min must be <= max`)
    }
    if (spec.type === 'categorical' && !spec.categories?.length) {
      errors.push(`${where}: a categorical column needs at least one category`)
    }
  })
  return errors
}

// -- queries ------------------------------------------------------------------

export function useProviderOptions() {
  return useQuery({
    queryKey: ['providers', 'options'],
    queryFn: () =>
      apiGet<{ providers: ProviderOption[] }>('/providers/options').then((r) => r.providers),
  })
}

// -- mutations ---------------------------------------------------------------

export function useSuggestColumns() {
  return useMutation({
    mutationFn: (body: { description: string; column_names?: string[] }) =>
      apiPost<{ specs: ColumnSpec[]; validation_errors: string[] }>(
        '/generate/suggest-columns',
        body,
      ),
  })
}

export function usePreview() {
  return useMutation({
    mutationFn: (body: GenerationBody & { preview_rows?: number }) =>
      apiPost<PreviewResult>('/generate/preview', { ...body, preview_rows: 5 }),
  })
}

export function useEstimate() {
  return useMutation({
    mutationFn: (body: GenerationBody) => apiPost<EstimateResult>('/generate/estimate', body),
  })
}

export function useStartRun() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: GenerationBody) =>
      apiPost<{ id: string }>('/generate/run', body).then((r) => r.id),
    onSuccess: () => {
      // a new Dataset Version is about to appear in the tree
      queryClient.invalidateQueries({ queryKey: ['dataset_versions'] })
    },
  })
}
