/**
 * Jobs: snapshots over SSE with a polling fallback (issue #11 defines the
 * stream contract; this consumes it).
 */

import { useEffect, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { apiGet } from '@/api/client'

export interface JobDto {
  id: string
  type: string
  project_id: string | null
  status: string
  resumable: boolean
  params: Record<string, unknown>
  progress: Record<string, unknown>
  result: Record<string, unknown> | null
  error: string | null
  created_at: string
}

export interface JobSnapshot {
  status: string
  progress: Record<string, unknown>
  result: Record<string, unknown> | null
  error: string | null
}

export const TERMINAL_STATUSES = new Set(['completed', 'failed', 'cancelled'])

export function jobPercent(progress: Record<string, unknown>): number {
  const done = Number(progress.done ?? 0)
  const total = Number(progress.total ?? 0)
  if (total > 0) return Math.min(100, Math.round((done / total) * 100))
  return Number(progress.pct ?? 0)
}

/** Poll the recent-jobs list; the banner shows everything not yet terminal. */
export function useRecentJobs(limit = 50) {
  return useQuery({
    queryKey: ['jobs', 'recent', limit],
    queryFn: () => apiGet<{ jobs: JobDto[] }>(`/jobs?limit=${limit}`).then((r) => r.jobs),
    refetchInterval: (query) =>
      (query.state.data ?? []).some((job) => !TERMINAL_STATUSES.has(job.status)) ? 1500 : 5000,
  })
}

/**
 * Live snapshot for one job: initial GET, then the SSE
 * `/api/jobs/{id}/events` stream. If the stream dies before a terminal
 * event, fall back to 1s polling until the job settles.
 */
export function useJobSnapshot(jobId: string | null) {
  const [snap, setSnap] = useState<JobSnapshot | null>(null)
  const [sseDown, setSseDown] = useState(false)

  const detail = useQuery({
    queryKey: ['job', jobId],
    queryFn: () => apiGet<JobDto>(`/jobs/${jobId}`),
    enabled: Boolean(jobId),
    refetchInterval: (query) => {
      if (!jobId) return false
      const status = snap?.status ?? query.state.data?.status
      if (status && TERMINAL_STATUSES.has(status)) return false
      return sseDown || query.state.status === 'error' ? 1000 : false
    },
  })

  useEffect(() => {
    setSnap(null)
    setSseDown(false)
    if (!jobId) return
    const EventSourceCtor = globalThis.EventSource
    if (!EventSourceCtor) {
      setSseDown(true)
      return
    }
    const es = new EventSourceCtor(`/api/jobs/${jobId}/events`)
    const onEvent = (event: MessageEvent) => {
      try {
        setSnap(JSON.parse(event.data) as JobSnapshot)
      } catch {
        // ignore malformed frames
      }
    }
    for (const name of ['progress', 'completed', 'failed', 'cancelled', 'interrupted']) {
      es.addEventListener(name, onEvent)
    }
    es.addEventListener('end', () => es.close())
    es.onerror = () => {
      es.close()
      setSseDown(true)
    }
    return () => es.close()
  }, [jobId])

  const fallback = detail.data
    ? {
        status: detail.data.status,
        progress: detail.data.progress,
        result: detail.data.result,
        error: detail.data.error,
      }
    : null
  return snap ?? fallback
}
