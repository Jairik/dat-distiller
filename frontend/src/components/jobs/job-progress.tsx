/**
 * Live progress for one job: SSE-backed bar + cancel, terminal states, and
 * a slot where callers render the run's result.
 */

import { useEffect, useRef } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { apiPost } from '@/api/client'
import { AnimatedNumber } from '@/components/motion'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { TERMINAL_STATUSES, jobPercent, useJobSnapshot } from '@/lib/jobs'

export function JobProgress({
  jobId,
  label = 'Job',
  renderResult,
  terminalExtra,
  onSettled,
  onStatus,
}: {
  jobId: string
  label?: string
  renderResult?: (result: Record<string, unknown>) => React.ReactNode
  terminalExtra?: React.ReactNode
  /**
   * Called once when the job first reaches a terminal state — the step needs it
   * to know a Dataset Version now exists. `result` is null unless it completed.
   */
  onSettled?: (result: Record<string, unknown> | null, status: string) => void
  /**
   * Called whenever the status changes. Lets a parent react to a status (say,
   * offer Resume for an interrupted run) without opening a *second* SSE
   * connection to the same job, which is what calling `useJobSnapshot` in the
   * parent would have done.
   */
  onStatus?: (status: string) => void
}) {
  const queryClient = useQueryClient()
  const { snapshot, error, retry } = useJobSnapshot(jobId)
  const cancel = useMutation({
    mutationFn: () => apiPost<unknown>(`/jobs/${jobId}/cancel`, {}),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['job', jobId] }),
  })
  const settled = useRef(false)
  useEffect(() => {
    settled.current = false
  }, [jobId])
  useEffect(() => {
    if (!snapshot || settled.current || !TERMINAL_STATUSES.has(snapshot.status)) return
    settled.current = true
    onSettled?.(snapshot.result, snapshot.status)
  }, [snapshot, onSettled])
  useEffect(() => {
    if (snapshot) onStatus?.(snapshot.status)
  }, [snapshot?.status, onStatus])

  if (!snapshot) {
    // The job's state is unknown, which is not the same as "still starting".
    // Saying "Waiting for …" here told someone to leave the page and come back
    // to a run that may not exist — a 500 on the job detail, a pruned row, or a
    // server that went away mid-run all looked identical, and identically forever.
    if (error) {
      return (
        <div role="alert" className="flex flex-col items-start gap-2 text-sm text-destructive">
          <p>{`This ${label.toLowerCase()} could not be read, so its state is unknown.`}</p>
          <p className="text-xs text-muted-foreground">{error.message}</p>
          <Button size="sm" variant="outline" onClick={retry}>
            Try again
          </Button>
        </div>
      )
    }
    return <p className="text-sm text-muted-foreground">Waiting for {label}…</p>
  }
  const percent = jobPercent(snapshot.progress)
  const running = !TERMINAL_STATUSES.has(snapshot.status)

  return (
    <div className="flex flex-col gap-2" data-testid={`job-${jobId}`}>
      <div className="flex items-center justify-between gap-3 text-sm">
        <span className="font-medium">
          {label}
          <span className="ml-2 text-muted-foreground">{snapshot.status}</span>
        </span>
        <span className="flex items-center gap-3">
          <AnimatedNumber value={percent} format={(v) => `${Math.round(v)}%`} className="tabular-nums" />
          {running && (
            <Button size="sm" variant="ghost" onClick={() => cancel.mutate()}>
              Cancel
            </Button>
          )}
        </span>
      </div>
      <Progress
        value={percent}
        aria-label={label}
        className={snapshot.status === 'failed' ? '[&>div]:bg-destructive' : undefined}
      />
      {snapshot.status === 'failed' && (
        <p role="alert" className="text-sm text-destructive">
          {snapshot.error ?? 'The job failed.'}
        </p>
      )}
      {snapshot.status === 'completed' && snapshot.result && renderResult?.(snapshot.result)}
      {!running && terminalExtra}
    </div>
  )
}
