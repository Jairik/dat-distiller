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
}) {
  const queryClient = useQueryClient()
  const snapshot = useJobSnapshot(jobId)
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

  if (!snapshot) {
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
