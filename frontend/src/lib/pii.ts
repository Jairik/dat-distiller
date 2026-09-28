/**
 * PII findings: what was found, and the three things you can do about it.
 *
 * The rule this module exists to enforce is that **raw PII never reaches the
 * browser**. The backend already masks every example it stores; this layer must
 * not un-mask it, re-derive it, or fall back to showing the underlying value if
 * a finding arrives without one. A finding with no examples says so.
 *
 * The three actions map to CONTEXT.md's vocabulary exactly: **Continue**
 * (acknowledge the Check and carry on), **Mask** (rewrite the detected spans in
 * a child Dataset Version), **Drop** (remove the column in a child Version).
 * Only the first does not create a new version.
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiGet, apiPost } from '@/api/client'

export type PiiAction = 'mask' | 'drop'

export const PII_ACTIONS: PiiAction[] = ['mask', 'drop']

export interface PiiFinding {
  column: string
  detector: string
  count: number
  /** Already masked by the backend (`j***.com`). Never raw. */
  example_cells: string[]
  row_examples: Array<number | string>
}

export interface PiiScan {
  version_id: string
  findings: PiiFinding[]
  summary: {
    columns: string[]
    detectors: Record<string, number>
    total_findings: number
  }
  actions: PiiAction[]
}

export interface PiiActionSummary {
  columns: Record<string, { action: string; replacements: number; note?: string }>
  masked: string[]
  dropped: string[]
  noops: string[]
}

export interface PiiActionResult {
  version: { id: string; parent_id: string | null; number: number; origin: string }
  summary: PiiActionSummary
  remaining_findings: { columns: string[]; detectors: Record<string, number>; total_findings: number }
}

export function usePiiScan(versionId: string | undefined) {
  return useQuery({
    queryKey: ['pii', versionId],
    queryFn: () => apiGet<PiiScan>(`/dataset-versions/${versionId}/pii`),
    enabled: Boolean(versionId),
    retry: false,
  })
}

export function useApplyPiiActions(versionId: string | undefined) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (actions: Record<string, PiiAction>) =>
      apiPost<PiiActionResult>(`/dataset-versions/${versionId}/pii/actions`, { actions }),
    onSuccess: () => {
      // mask and drop both produce a new Dataset Version, and they change what
      // a scan would find, so everything PII-related is re-read
      queryClient.invalidateQueries({ queryKey: ['pii'] })
      queryClient.invalidateQueries({ queryKey: ['checks'] })
      queryClient.invalidateQueries({ queryKey: ['dataset_versions'] })
    },
  })
}

/** What a Check's details say about a column, for the warn path. */
export interface PiiCheckDetails {
  column?: string
  detector?: string
  detectors?: Record<string, number>
  count?: number
  examples?: string[]
  row_examples?: Array<number | string>
}

/** Read a `pii_found` Check's details defensively — they are not ours alone. */
export function piiDetailsOf(details: unknown): PiiCheckDetails | null {
  if (typeof details !== 'object' || details === null) return null
  const record = details as Record<string, unknown>
  const column = typeof record.column === 'string' ? record.column : null
  if (!column) return null
  return {
    column,
    detector: typeof record.detector === 'string' ? record.detector : undefined,
    detectors:
      typeof record.detectors === 'object' && record.detectors !== null
        ? (record.detectors as Record<string, number>)
        : undefined,
    count: typeof record.count === 'number' ? record.count : undefined,
    examples: Array.isArray(record.examples) ? record.examples.map(String) : [],
    row_examples: Array.isArray(record.row_examples) ? record.row_examples : [],
  }
}

/**
 * Every example a finding will ever show, as strings.
 *
 * Examples are rendered as the backend produced them. If a finding somehow
 * arrived with no examples, the UI shows nothing rather than reaching back to
 * the data for the real value — that fallback is exactly how raw PII would end
 * up on screen.
 */
export function examplesOf(finding: PiiFinding): string[] {
  return finding.example_cells.filter((cell) => typeof cell === 'string' && cell.length > 0)
}
