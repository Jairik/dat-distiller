/**
 * Checks + Acknowledgements: the responsible-use safeguards raised at the end
 * of a step (PII found, near-copies, class imbalance, leakage, fairness gaps).
 *
 * A Check never blocks: a `warning` just needs an explicit Acknowledgement
 * before the step that raised it counts as complete. `unacknowledged_warnings`
 * is the gate the API reports and the count `ChecksPanel` waits on.
 *
 * Checks belong to a *subject* — a Project, a Dataset Version or a Training
 * Run — so the same panel is reused at the end of every step.
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiGet, apiPost } from '@/api/client'

export type CheckSeverity = 'info' | 'warning'

/** One Check as the API returns it (`Check.to_dict()` on the backend). */
export interface Check {
  id: string
  kind: string
  severity: CheckSeverity
  message: string
  subject_type: string
  subject_id: string
  details: Record<string, unknown>
  acknowledged: boolean
  acknowledged_at?: string | null
  note?: string | null
}

export interface ChecksResponse {
  checks: Check[]
  unacknowledged_warnings: number
}

/** Query key for one subject's Checks; `['checks']` covers every subject. */
export function checksKey(subjectType: string, subjectId: string | undefined) {
  return ['checks', subjectType, subjectId] as const
}

/** Checks for one subject, plus the unacknowledged-warning count for the gate. */
export function useChecks(subjectType: string, subjectId: string | undefined) {
  return useQuery({
    queryKey: checksKey(subjectType, subjectId),
    queryFn: () =>
      apiGet<ChecksResponse>(
        `/checks?subject_type=${encodeURIComponent(subjectType)}&subject_id=${encodeURIComponent(subjectId ?? '')}`,
      ),
    enabled: Boolean(subjectId),
  })
}

/** Convenience for headers and banners that only show a warning count. */
export function useUnacknowledgedWarningCount(
  subjectType: string,
  subjectId: string | undefined,
): number {
  const query = useChecks(subjectType, subjectId)
  return query.data?.unacknowledged_warnings ?? 0
}

export interface AcknowledgeVars {
  checkId: string
  /** Optional reason for continuing; stored with the Acknowledgement for Cards. */
  note?: string
}

/**
 * Records an Acknowledgement for one Check (`POST /checks/{id}/acknowledge`).
 * The note is optional and sent as `null` when left blank; every checks query
 * is invalidated on success so panels and warning badges re-read the gate.
 */
export function useAcknowledgeCheck() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ checkId, note }: AcknowledgeVars) =>
      apiPost<Check>(`/checks/${checkId}/acknowledge`, { note: note?.trim() ? note.trim() : null }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['checks'] }),
  })
}
