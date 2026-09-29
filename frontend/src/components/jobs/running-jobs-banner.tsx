/**
 * Sticky "work happening elsewhere" bar: any queued/running job in any
 * project, visible from every page; click one to jump to its project.
 */

import { Link } from 'react-router-dom'
import { motion } from 'motion/react'
import { jobPercent, TERMINAL_STATUSES, useRecentJobs } from '@/lib/jobs'

export function RunningJobsBanner() {
  const jobs = useRecentJobs()
  const active = (jobs.data ?? []).filter((job) => !TERMINAL_STATUSES.has(job.status))
  if (active.length === 0) return null

  return (
    <motion.div
      initial={{ y: -16, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      className="sticky top-0 z-40 border-b border-border bg-background/95 px-6 py-2 backdrop-blur"
    >
      <div className="mx-auto flex max-w-5xl flex-wrap items-center gap-x-6 gap-y-1">
        {/* `role="status"` is for a message that changes once, and this banner is
            polled every 1.5s: on the wrapper it re-announced the whole thing,
            percentage included, about 40 times a minute. Only the count is news;
            the progress is a meter, and it carries a title for anyone who wants
            to read it. */}
        <span
          role="status"
          className="text-xs font-medium uppercase tracking-wide text-muted-foreground"
        >
          {active.length} job{active.length === 1 ? '' : 's'} running
        </span>
        {active.map((job) => {
          const percent = jobPercent(job.progress)
          const body = (
            <span title={`${job.type.replace(/_/g, ' ')} — ${percent}%`}>
              <span className="capitalize">{job.type.replace(/_/g, ' ')}</span>
              <span className="text-muted-foreground">{percent}%</span>
            </span>
          )
          return job.project_id ? (
            <Link
              key={job.id}
              to={`/projects/${job.project_id}/dataset`}
              className="flex items-center gap-2 text-sm hover:underline"
            >
              {body}
            </Link>
          ) : (
            <span key={job.id} className="flex items-center gap-2 text-sm">
              {body}
            </span>
          )
        })}
      </div>
    </motion.div>
  )
}
