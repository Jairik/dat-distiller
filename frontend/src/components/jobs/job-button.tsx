/**
 * One-button job launcher: start on click, inline live progress while the
 * job runs, result inline when it completes, then run again.
 */

import { useState } from 'react'
import { HoverPress } from '@/components/motion'
import { Button } from '@/components/ui/button'
import { JobProgress } from '@/components/jobs/job-progress'

export function JobButton({
  label,
  start,
  resultLabel = 'Result',
  renderResult,
}: {
  label: string
  start: () => Promise<{ id: string }>
  resultLabel?: string
  renderResult?: (result: Record<string, unknown>) => React.ReactNode
}) {
  const [jobId, setJobId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  if (jobId) {
    return (
      <div className="w-full">
        <JobProgress
          jobId={jobId}
          label={label}
          renderResult={(result) => (
            <div className="rounded-md border border-border bg-muted/40 p-3 text-sm">
              <span className="font-medium">{resultLabel}: </span>
              {renderResult?.(result) ?? JSON.stringify(result)}
            </div>
          )}
          terminalExtra={
            <Button size="sm" variant="secondary" className="w-fit" onClick={() => setJobId(null)}>
              Run again
            </Button>
          }
        />
      </div>
    )
  }

  return (
    <div className="flex flex-col items-start gap-2">
      <HoverPress>
        <Button
          onClick={async () => {
            setError(null)
            try {
              const job = await start()
              setJobId(job.id)
            } catch (exc) {
              setError((exc as Error).message)
            }
          }}
        >
          {label}
        </Button>
      </HoverPress>
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
    </div>
  )
}
