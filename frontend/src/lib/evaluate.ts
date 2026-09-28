/**
 * On-demand evaluation, prediction and bundle download.
 *
 * The rule this module encodes is that **nothing here runs until it is asked
 * for**. There is no prefetch, no `useEffect` that fires a plot request, and the
 * plot cache is keyed by (run, model, plot) so switching Models and back does
 * not refetch. A user who opens a leaderboard pays for nothing until they press
 * a button.
 *
 * A plot that cannot be produced — a ROC curve for a Model with no class
 * probabilities — comes back from the backend as `200 {error: "..."}` rather
 * than a 500, precisely so the UI can show that one card's reason without
 * losing the rest of the panel.
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import { apiGet, apiPost } from '@/api/client'

export type PlotName =
  | 'confusion_matrix'
  | 'roc'
  | 'precision_recall'
  | 'residuals'
  | 'feature_importance'

export const CLASSIFICATION_PLOTS: PlotName[] = [
  'confusion_matrix',
  'roc',
  'precision_recall',
]

export const REGRESSION_PLOTS: PlotName[] = ['residuals']

/** Every Model can report feature importance, whatever its Task Type. */
export const ALL_PLOTS: PlotName[] = [...CLASSIFICATION_PLOTS, ...REGRESSION_PLOTS]

export const PLOT_LABELS: Record<PlotName, string> = {
  confusion_matrix: 'Confusion matrix',
  roc: 'ROC curve',
  precision_recall: 'Precision / recall',
  residuals: 'Residuals',
  feature_importance: 'Feature importance',
}

export const PLOT_HINTS: Record<PlotName, string> = {
  confusion_matrix: 'How often each answer was confused with each other.',
  roc: 'Trade-off between catching the positive class and raising false alarms.',
  precision_recall: 'The same trade-off, weighted towards what you actually act on.',
  residuals: 'Actual minus predicted, against what was predicted.',
  feature_importance: 'How much each column is worth, by shuffling it.',
}

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

export interface ConfusionMatrix {
  labels: string[]
  matrix: number[][]
  row_normalised: number[][]
  support: number[]
  n: number
}

export interface RocCurve {
  kind: 'binary' | 'multiclass_ovr'
  positive_class?: string
  fpr?: number[]
  tpr?: number[]
  thresholds?: number[]
  auc: number | null
  classes?: string[]
  curves?: Record<string, { fpr: number[]; tpr: number[] }>
  auc_macro?: number
}

export interface PrecisionRecall {
  kind: 'binary' | 'multiclass_ovr'
  positive_class?: string
  precision?: number[]
  recall?: number[]
  average_precision: number | null
  classes?: string[]
  curves?: Record<string, { precision: number[]; recall: number[] }>
  average_precision_macro?: number
}

export interface Residuals {
  residuals: number[]
  predicted: number[]
  actual: number[]
  mae: number
  mean_residual: number
  n: number
}

export interface FeatureImportance {
  method: string
  baseline: number
  repeats: number
  importance: Array<{ feature: string; drop: number }>
}

export type PlotResult =
  | ({ plot: 'confusion_matrix' } & ConfusionMatrix)
  | ({ plot: 'roc' } & RocCurve)
  | ({ plot: 'precision_recall' } & PrecisionRecall)
  | ({ plot: 'residuals' } & Residuals)
  | ({ plot: 'feature_importance' } & FeatureImportance)

/**
 * One plot, fetched only when `enabled` is true.
 *
 * `staleTime: Infinity` matters as much as `enabled`: without it React Query
 * would refetch on every remount, and clicking between two Models and back
 * would silently re-run a search that costs real time.
 */
export function usePlot(
  runId: string | undefined,
  model: string | undefined,
  plot: PlotName | undefined,
  enabled: boolean,
) {
  return useQuery({
    queryKey: ['train-plot', runId, model, plot],
    queryFn: () =>
      apiGet<PlotResult & { error?: string | null }>(
        `/train/runs/${runId}/plots?model=${model}&plot=${plot}`,
      ),
    enabled: enabled && Boolean(runId && model && plot),
    staleTime: Infinity,
    retry: false,
  })
}

export function invalidatePlots(client: ReturnType<typeof useQueryClient>) {
  client.invalidateQueries({ queryKey: ['train-plot'] })
}

export interface PredictionResult {
  run_id: string
  model: string
  rows: number
  columns: string[]
  predictions: Array<Record<string, unknown>>
  download_url: string
}

export function usePredict() {
  return useMutation({
    mutationFn: ({
      runId,
      model,
      file,
    }: {
      runId: string
      model: string
      file: File
    }) => {
      const form = new FormData()
      form.append('file', file)
      return apiPost<PredictionResult>(
        `/train/runs/${runId}/predict?model=${encodeURIComponent(model)}`,
        form,
      )
    },
  })
}

export function bundleUrl(runId: string, model: string): string {
  return `/api/train/runs/${runId}/bundle?model=${encodeURIComponent(model)}`
}

// -- reading the leaderboard --------------------------------------------------

/** Metrics every entry on this board actually carries a value for. */
export function rankableMetrics(board: LeaderboardEntry[]): string[] {
  const seen = new Map<string, boolean>()
  for (const entry of board) {
    if (entry.status !== 'ok') continue
    for (const [name, value] of Object.entries(entry.metrics ?? {})) {
      if (name === 'confusion_matrix') continue
      if (value?.value !== null && value?.value !== undefined) seen.set(name, true)
    }
  }
  return [...seen.keys()]
}

export interface RankedEntry extends LeaderboardEntry {
  displayRank: number | null
  rankable: boolean
  /** Why this Model could not be ranked, in words. */
  unrankedReason: string | null
}

/**
 * Re-rank the board by a chosen metric.
 *
 * The backend's `rank` is competition ranking on its own primary metric. When
 * the user picks a different one, ties are re-ranked the same way (1, 1, 3) and a
 * Model with no value for that metric is **left unranked with its reason** —
 * sorting it arbitrarily would put an unmeasurable Model above a measured one,
 * which is the exact thing a leaderboard must never do.
 */
export function rankBy(
  board: LeaderboardEntry[],
  metric: string,
  higherIsBetter: boolean,
): RankedEntry[] {
  const scored = board.map((entry) => {
    // A Model that did not fit has no score, whatever a stale payload claims.
    // Ranking it would put a number on the board that was never measured.
    if (entry.status !== 'ok') {
      return { entry, value: null as number | null, reason: null as string | null }
    }
    const value = metric === entry.primary_metric ? entry.primary?.value : entry.metrics?.[metric]?.value
    const reason =
      metric === entry.primary_metric
        ? (entry.primary?.reason ?? null)
        : (entry.metrics?.[metric]?.reason ?? null)
    return { entry, value, reason }
  })

  const comparable = scored
    .filter((s) => s.value !== null && s.value !== undefined)
    .sort((a, b) =>
      higherIsBetter ? (b.value as number) - (a.value as number) : (a.value as number) - (b.value as number),
    )

  const ranks = new Map<string, number>()
  let last: number | null = null
  let lastValue: number | null = null
  comparable.forEach((s, index) => {
    const value = s.value as number
    // competition ranking: a tie shares a rank and the next Model is skipped
    if (lastValue !== null && value === lastValue) {
      ranks.set(s.entry.model, last as number)
    } else {
      last = index + 1
      lastValue = value
      ranks.set(s.entry.model, last)
    }
  })

  return board
    .map((entry) => {
      const found = scored.find((s) => s.entry.model === entry.model)
      const rankable = ranks.has(entry.model)
      return {
        ...entry,
        displayRank: ranks.get(entry.model) ?? null,
        rankable,
        unrankedReason: rankable
          ? null
          : entry.status === 'failed'
            ? (entry.error ?? 'this Model did not fit')
            : (found?.reason ?? `no ${metric} was measured for this Model`),
      }
    })
    // Ranked Models first, best first; the unmeasurable ones after, in the order
    // the board already had them. Returning board order here would make
    // re-ranking a cosmetic change, which is the whole thing it is not.
    .sort((a, b) => {
      if (a.displayRank !== null && b.displayRank !== null) return a.displayRank - b.displayRank
      if (a.displayRank !== null) return -1
      if (b.displayRank !== null) return 1
      return 0
    })
}

/** Did the run compute this metric at all, whatever any single Model scored? */
export function metricAvailable(
  board: LeaderboardEntry[],
  metric: string,
): boolean {
  return board.some(
    (entry) => entry.metrics?.[metric]?.value !== null && entry.metrics?.[metric]?.value !== undefined,
  )
}
