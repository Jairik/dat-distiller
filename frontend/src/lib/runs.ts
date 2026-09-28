/** Training Runs for a Project, and the Models they fitted. */

import { useQuery } from '@tanstack/react-query'

import { apiGet } from '@/api/client'

export interface TrainingRunSummary {
  training_run_id: string
  job_id: string
  project_id: string
  status: string
  progress: Record<string, unknown>
  created_at: string
  finished_at: string | null
  version_id: string
  target: string
  task_type: string
  seed: number
  primary_metric: string
  n_models: number
  best_model: string | null
  best_primary_value: number | null
  warnings: string[]
  error: string | null
}

export interface TrainingRunsResponse {
  project_id: string
  count: number
  runs: TrainingRunSummary[]
}

export function useTrainingRuns(projectId: string | undefined) {
  return useQuery({
    queryKey: ['training-runs', projectId],
    queryFn: () =>
      apiGet<TrainingRunsResponse>(`/train/runs?project_id=${projectId}`),
    enabled: Boolean(projectId),
  })
}

/**
 * The primary metric's value, formatted.
 *
 * `null` is a real answer here — a Model with no comparable metric could not be
 * ranked — so it reads as an em dash rather than a zero, which would look like a
 * genuinely terrible score.
 */
export function formatPrimary(
  value: number | null,
  higherIsBetter = true,
): string {
  if (value === null || value === undefined) return '—'
  return `${value.toFixed(4)}${higherIsBetter ? ' ↑' : ' ↓'}`
}
