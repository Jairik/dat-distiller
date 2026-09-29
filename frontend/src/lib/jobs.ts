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

/**
 * Statuses a job will never move on from by itself.
 *
 * `interrupted` belongs here: a server restart stops a run dead, and nothing
 * will resume it but the user. Treating it as still-running left the progress
 * bar spinning forever, `onSettled` never firing, and — the real cost — no
 * Resume button, so an interrupted Labeling run was unrecoverable from the UI.
 * Resuming re-queues the job, so the snapshot picks it up again from the start.
 */
export const TERMINAL_STATUSES = new Set([
  'completed',
  'failed',
  'cancelled',
  'interrupted',
])

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
    // No automatic retries: one failed read is reported straight away rather
    // than after a backoff the person waits through to learn the same thing.
    // Recovery is the `refetchInterval` below, which backs off and eventually
    // gives up so a job that cannot be read is not requested forever.
    retry: false,
    refetchInterval: (query) => {
      if (!jobId) return false
      const status = snap?.status ?? query.state.data?.status
      if (status && TERMINAL_STATUSES.has(status)) return false
      if (sseDown) return 1000
      if (query.state.status === 'error') {
        const failures = query.state.errorUpdateCount ?? 0
        return failures >= 4 ? false : 1000 * 2 ** Math.min(failures, 3)
      }
      return false
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
  // With no snapshot *and* a failed read, the job's state is unknown — not
  // "still starting". Returning only the snapshot made that distinction
  // impossible, so the caller rendered "Waiting for …" forever with no error and
  // no way to tell a run that is happening from one that died.
  return {
    snapshot: snap ?? fallback,
    error: snap ? null : (detail.error ?? null),
    retry: () => void detail.refetch(),
  }
}
