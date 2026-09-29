/**
 * The Fairness Report's data layer.
 *
 * Two things this module refuses to do, because both are how a fairness report
 * misleads:
 *
 * **It never smooths over an unmeasured group.** The backend returns `null` with
 * a reason for a group it could not measure, and this layer passes that through
 * to the UI rather than dropping the group. A two-group report that silently
 * omitted the third group reads as "these two are equally treated", which is not
 * what the data says.
 *
 * **It never hides a gap that a threshold cannot judge.** Regression MAE is in
 * the Target's own units, so `comparable: false` — the *relative* gap is what a
 * threshold can act on. Showing a raw unit difference against a rate threshold
 * would flag things that are not findings and hide things that are.
 */

import { useMutation } from '@tanstack/react-query'

import { apiPost } from '@/api/client'

export interface GroupMetric {
  value: number | null
  reason: string | null
}

export interface FairnessGroup {
  group: string
  n_version: number
  n_total: number
  n_train: number
  n_test: number
  n_actual_positive: number | null
  n_actual_negative: number | null
  n_predicted_positive: number | null
  class_counts: Record<string, number>
  measured: boolean
  reason: string | null
  metrics: Record<string, GroupMetric>
}

export interface FairnessGap {
  name: string
  label: string
  metric: string
  unit: string
  description: string
  threshold: number
  groups_total: number
  groups_compared: number
  groups_unmeasured: string[]
  value: number | null
  reason: string | null
  comparable: boolean
  exceeds: boolean | null
  is_widest: boolean
}

export interface FairnessReport {
  training_run_id: string
  project_id: string
  version_id: string
  target: string
  task_type: 'classification' | 'regression'
  model: string
  model_label: string
  seed: number | null
  /** The Model is refit for the report; it was not the object the run kept. */
  refit: boolean
  sensitive_attribute: string
  declared_sensitive_attribute: string | null
  sensitive_attribute_excluded_from_features: boolean
  positive_label: string | null
  positive_class: string | null
  classes: string[]
  threshold: number
  threshold_source: string
  split: { train_rows: number; test_rows: number; reused_stored_split: boolean }
  overall: Record<string, GroupMetric>
  n_groups: number
  n_groups_measured: number
  groups: FairnessGroup[]
  unmeasured_groups: Array<{ group: string; reason: string; n_test: number }>
  gaps: FairnessGap[]
  gaps_exceeding_threshold: string[]
  largest_gap: FairnessGap | null
  largest_comparable_gap: FairnessGap | null
  gaps_within_threshold: string[]
  gaps_not_compared: string[]
  notes: string[]
  checks_raised: Check[]
  check_step: string
  unacknowledged_warnings: Check[]
}

export interface Check {
  id: string
  kind: string
  severity: string
  message: string
  subject_type: string
  subject_id: string
  details: Record<string, unknown>
  acknowledged: boolean
  acknowledged_at: string | null
  note: string | null
}

export interface FairnessRequest {
  runId: string
  sensitive_attribute: string
  model?: string
  positive_label?: string | null
  gap_threshold?: number
  raise_check?: boolean
}

export function useFairnessReport() {
  return useMutation({
    mutationFn: ({ runId, ...body }: FairnessRequest) =>
      apiPost<FairnessReport>(`/train/runs/${runId}/fairness`, body),
  })
}

/** The metrics a group's bars are drawn for, in the order they should be read. */
export const METRIC_ORDER = [
  'accuracy',
  'tpr',
  'fpr',
  'selection_rate',
  'mae',
  'mae_relative',
] as const

export const METRIC_LABELS: Record<string, string> = {
  accuracy: 'Accuracy',
  tpr: 'True positive rate',
  fpr: 'False positive rate',
  selection_rate: 'Selection rate',
  mae: 'Mean absolute error',
  mae_relative: 'Mean absolute error (relative)',
}

export const METRIC_PLAIN: Record<string, string> = {
  accuracy: 'how often the Model was right, overall for that group',
  tpr: 'of the rows that should have been flagged, how many it caught',
  fpr: 'of the rows that should not have been flagged, how many it flagged anyway',
  selection_rate: 'how often the group was flagged at all',
  mae: 'how far off the Model was, in the Target’s own units',
  mae_relative: 'how far off it was, relative to its overall error',
}

/**
 * One sentence on what a gap means, in words rather than symbols.
 *
 * A fairness report that only says "TPR gap 0.31" has told a reader a number
 * and nothing else. Each of these says what the two ends of the gap *are*.
 */
export function explainGap(gap: FairnessGap, report: FairnessReport): string {
  if (gap.value === null) {
    return `${gap.label} could not be compared: ${gap.reason ?? 'not enough groups had a value'}.`
  }
  const groups = report.groups
    // a group that does not carry the metric at all is not the same as one that
    // carries it as null: filtering with `?.value !== null` lets `undefined`
    // through and then reads `.value` off nothing
    .map((g) => ({ group: g.group, value: g.measured ? g.metrics[gap.metric]?.value : null }))
    .filter((entry): entry is { group: string; value: number } => entry.value !== null && entry.value !== undefined)
    .sort((a, b) => b.value - a.value)
  // `groups` is sorted descending, so the first is the *largest* value and the
  // last the smallest. For fpr or mae the largest is the group treated worst, so
  // the old `best`/`worst` naming put the worse group first in the sentence:
  // "False positive rate: str:a (0.60) against str:b (0.10)" named the group with
  // six times the false alarms first. Named for what they are instead of for a
  // good and a bad score, which is true for every metric and needs no
  // per-metric wording.
  const highest = groups[0]
  const lowest = groups.at(-1)
  if (!highest || !lowest || highest.group === lowest.group) {
    return `${gap.label} is ${formatGap(gap, report)} — only one group had a value.`
  }
  const base = `${highest.group} had the highest ${gap.metric.replace(/_/g, ' ')} at ${highest.value.toFixed(2)}, against ${lowest.group} at ${lowest.value.toFixed(2)}`
  const verdict =
    gap.exceeds === true
      ? `That is ${gap.value.toFixed(3)}, over the ${gap.threshold} threshold — flagged.`
      : gap.exceeds === false
        ? `That is ${gap.value.toFixed(3)}, within the ${gap.threshold} threshold.`
        : `A threshold of ${gap.threshold} cannot judge a gap in ${gap.unit}.`
  return `${gap.label}: ${base}. ${verdict}`
}

export function formatGap(gap: FairnessGap, _report: FairnessReport): string {
  if (gap.value === null) return 'not comparable'
  return `${gap.value.toFixed(3)}${gap.comparable ? '' : ` ${gap.unit}`}`
}

/** The gap to point the reader at first. */
export function headlineGap(report: FairnessReport): FairnessGap | null {
  return report.largest_comparable_gap ?? report.largest_gap
}
