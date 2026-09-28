/**
 * The Train step's data layer.
 *
 * Two rules shape everything here.
 *
 * **The Plan is authoritative.** `/train/plan` is what the backend will actually
 * do — resolved features, the Task Type, the split sizes, which Models are
 * usable, the metrics available. The step asks before it lets you run, so a
 * request that would be refused comes back as a sentence you can read rather than
 * a 422 after you clicked Train.
 *
 * **The test split is the backend's business, not a knob.** The step never sends
 * a split, a test fraction, or a seed it invented; it sends the Target, the
 * features, and the Models. Everything else is derived and shown read-only, so
 * what the plan says is what the run will do.
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import { apiGet, apiPost } from '@/api/client'

// -- the registry -------------------------------------------------------------

export interface ModelSpecDto {
  name: string
  label: string
  library: string
  extra: string | null
  task_types: string[]
  supports_proba: boolean
  notes: string
  available: boolean
  class_path?: Record<string, string>
  default_hyperparameters?: Record<string, Record<string, unknown>>
  search_space?: Record<string, unknown[]>
}

export interface ModelsResponse {
  models: ModelSpecDto[]
  extras: Record<string, boolean>
  library_versions: Record<string, string | null>
}

export interface MetricSpecDto {
  name: string
  label: string
  task_type: string
  higher_is_better: boolean
  needs_proba: boolean
  is_default: boolean
  description: string
}

export interface MetricsResponse {
  task_types: Record<string, { metrics: MetricSpecDto[]; default_primary_metric: string }>
}

export function useModels() {
  return useQuery({
    queryKey: ['train-models'],
    queryFn: () => apiGet<ModelsResponse>('/train/models'),
    staleTime: 5 * 60 * 1000,
  })
}

export function useMetrics(taskType: string | undefined) {
  return useQuery({
    queryKey: ['train-metrics', taskType],
    queryFn: () => apiGet<MetricsResponse>(`/train/metrics?task_type=${taskType}`),
    enabled: Boolean(taskType),
  })
}

// -- the request --------------------------------------------------------------

export interface TrainBody {
  version_id: string
  target: string
  task_type?: 'classification' | 'regression' | null
  features?: string[] | null
  exclude_features?: string[]
  sensitive_attribute?: string | null
  include_sensitive_attribute?: boolean
  exclude_unreviewed?: boolean
  models?: string[] | null
  primary_metric?: string | null
  tuning?: {
    enabled: boolean
    strategy: 'random' | 'grid'
    n_iter: number
    cv: number
    metric?: string | null
  }
}

export function usePlan() {
  return useMutation({
    mutationFn: (body: TrainBody) => apiPost<TrainPlan>('/train/plan', body),
  })
}

export function useStartTraining() {
  return useMutation({
    mutationFn: (body: TrainBody) => apiPost<{ id: string }>('/train/run', body).then((r) => r.id),
  })
}

export function useTrainingRun(runId: string | null | undefined) {
  return useQuery({
    queryKey: ['train-run', runId],
    queryFn: () => apiGet<TrainingRun>(`/train/runs/${runId}`),
    enabled: Boolean(runId),
  })
}

export function invalidateTraining(client: ReturnType<typeof useQueryClient>) {
  client.invalidateQueries({ queryKey: ['train-runs'] })
  client.invalidateQueries({ queryKey: ['train-run'] })
  client.invalidateQueries({ queryKey: ['checks'] })
}

// -- what comes back ----------------------------------------------------------

export interface MetricValue {
  value: number | null
  reason: string | null
}

export interface LeaderboardEntry {
  rank: number | null
  model: string
  label: string
  library: string
  task_type: string
  status: 'ok' | 'failed'
  error?: string | null
  classes: string[]
  hyperparameters: Record<string, unknown>
  fit_seconds: number
  metrics: Record<string, MetricValue>
  primary_metric: string
  primary: MetricValue
}

export interface TrainPlan {
  version_id: string
  target: string
  task_type: 'classification' | 'regression'
  label_family: string
  seed: number
  rows: number
  kept_rows: number
  train_rows: number
  test_rows: number
  n_source_features: number
  models: ModelSpecDto[]
  model_names: string[]
  unavailable_models: Array<{ model: string; reason: string }>
  primary_metric: string
  primary_metric_label: string
  primary_metric_higher_is_better: boolean
  metrics_available: string[]
  tuning: {
    enabled: boolean
    strategy: string
    n_iter: number
    cv: number
    metric?: string | null
    tunable: Record<string, boolean> | boolean
  }
  hyperparameters: Record<string, Record<string, unknown>>
}

export interface TrainingRun {
  training_run_id: string
  job_id?: string
  project_id: string
  version_id: string
  status: string
  progress: Record<string, unknown>
  created_at: string
  finished_at: string | null
  error: string | null
  target?: string
  task_type?: string
  seed?: number
  primary_metric?: string
  primary_metric_label?: string
  primary_metric_higher_is_better?: boolean
  models?: string[]
  leaderboard?: LeaderboardEntry[]
  training_split?: { rows?: number; n_samples?: number; n_features?: number; classes?: string[] }
  test_split?: { rows?: number; indices?: number[]; stratified?: boolean; test_size?: number }
  feature_names?: string[]
  setup?: Record<string, unknown>
  warnings?: string[]
  checks?: Array<Record<string, unknown>>
  split_indices?: { train: number[]; test: number[] }
}

/** Models a given Task Type can use, split into runnable and not. */
export function modelsForTaskType(
  models: ModelSpecDto[],
  taskType: string | undefined,
): { usable: ModelSpecDto[]; unavailable: ModelSpecDto[] } {
  const forTask = taskType
    ? models.filter((m) => m.task_types.includes(taskType))
    : models
  return {
    usable: forTask.filter((m) => m.available),
    unavailable: forTask.filter((m) => !m.available),
  }
}

/**
 * A one-line reason a Model cannot be used here.
 *
 * A model the user cannot run must say *why* and what to do, otherwise it just
 * looks broken.
 */
export function unavailableReason(model: ModelSpecDto): string {
  if (model.available) return ''
  return model.extra
    ? `needs the optional extra '${model.extra}' — uv sync --extra ${model.extra}`
    : 'not available in this install'
}
