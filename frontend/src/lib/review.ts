/**
 * The Review Queue's data layer.
 *
 * Decisions are held locally and applied in one call, because a reviewer
 * working down a list should be able to accept a dozen rows and commit once.
 * The threshold is a query parameter, not a saved setting: the slider is for
 * *looking* at a queue at a different bar, and changing what you are shown
 * must never silently change what gets trained on.
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiGet, apiPost } from '@/api/client'

export type Decision = 'accept' | 'override' | 'exclude'

export const DECISIONS: Decision[] = ['accept', 'override', 'exclude']

export interface QueueItem {
  row_index: number
  family: string
  question_type: string
  answer: unknown
  confidence: number | null
  decision: Decision | null
  override: unknown
  note: string | null
}

export interface Unlabeled {
  row_index: number
  family: string
  question_type: string
}

export interface ReviewQueue {
  version_id: string
  threshold: number
  families: Record<string, string>
  items: QueueItem[]
  unlabeled: Unlabeled[]
  /** Total outstanding across the whole version, not just this page. */
  queued_count: number
  returned_count: number
  unreviewed_count: number
  decisions: Decision[]
}

export interface ReviewStatus {
  version_id: string
  threshold: number
  queued_count: number
  unlabeled_count: number
  unreviewed_count: number
}

export interface ReviewRequest {
  decisions: Array<{
    row_index: number
    family: string
    question_type: string
    decision: Decision
    override?: unknown
    note?: string | null
  }>
  threshold?: number
  raise_check?: boolean
}

export interface ReviewResult {
  version: { id: string; parent_id: string | null; number: number; origin: string }
  outcome: { accepted: number; overridden: number; excluded_rows: number[]; excluded_count: number }
  unreviewed_count: number
  checks_raised: unknown[]
}

export function useReviewQueue(versionId: string | undefined, threshold: number) {
  return useQuery({
    queryKey: ['review-queue', versionId, threshold],
    queryFn: () =>
      apiGet<ReviewQueue>(
        `/dataset-versions/${versionId}/review-queue?threshold=${threshold}&limit=200`,
      ),
    enabled: Boolean(versionId),
  })
}

export function useReviewStatus(versionId: string | undefined, threshold: number) {
  return useQuery({
    queryKey: ['review-status', versionId, threshold],
    queryFn: () =>
      apiGet<ReviewStatus>(`/dataset-versions/${versionId}/review-status?threshold=${threshold}`),
    enabled: Boolean(versionId),
  })
}

export function useApplyReview(versionId: string | undefined) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: ReviewRequest) =>
      apiPost<ReviewResult>(`/dataset-versions/${versionId}/review`, body),
    onSuccess: (result) => {
      // a new Dataset Version now exists: the history panel, the queue and the
      // status all have to re-read
      queryClient.invalidateQueries({ queryKey: ['dataset_versions'] })
      queryClient.invalidateQueries({ queryKey: ['review-queue'] })
      queryClient.invalidateQueries({ queryKey: ['review-status'] })
      queryClient.invalidateQueries({ queryKey: ['checks'] })
      void result
    },
  })
}

/** Is this override value usable for that question type? */
export function overrideProblem(
  questionType: string,
  value: unknown,
): string | null {
  if (value === null || value === undefined || String(value).trim() === '') {
    return 'an override needs a value'
  }
  if (questionType === 'noul' && !['yes', 'no', 'true', 'false'].includes(String(value).toLowerCase())) {
    return 'a Noul is answered yes or no'
  }
  if (questionType === 'score' && !Number.isFinite(Number(value))) {
    return 'a Score is overridden with a number on the scale'
  }
  return null
}
