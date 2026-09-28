/**
 * Provenance breakdown as a tiny stacked bar: uploaded · synthetic · jev ·
 * human-reviewed (the four BREAKDOWN_BUCKETS the backend reports).
 */

import { cn } from '@/lib/utils'

export const PROVENANCE_COLORS: Record<string, string> = {
  uploaded: 'bg-sky-500',
  synthetic: 'bg-violet-500',
  jev: 'bg-amber-500',
  human_reviewed: 'bg-emerald-500',
}

const LABELS: Record<string, string> = {
  uploaded: 'Uploaded',
  synthetic: 'Synthetic',
  jev: 'Jev-labeled',
  human_reviewed: 'Human-reviewed',
}

export function ProvenanceBar({
  summary,
  className,
}: {
  summary: Record<string, number>
  className?: string
}) {
  const buckets = Object.keys(PROVENANCE_COLORS).filter((key) => (summary[key] ?? 0) > 0)
  const total = buckets.reduce((sum, key) => sum + (summary[key] ?? 0), 0)
  if (total === 0) {
    return <div className={cn('h-1.5 w-full rounded bg-muted', className)} aria-label="no rows" />
  }
  return (
    <div
      className={cn('flex h-1.5 w-full overflow-hidden rounded bg-muted', className)}
      role="img"
      aria-label={buckets.map((key) => `${LABELS[key]} ${summary[key]}`).join(', ')}
    >
      {buckets.map((key) => (
        <div
          key={key}
          className={PROVENANCE_COLORS[key]}
          style={{ width: `${((summary[key] ?? 0) / total) * 100}%` }}
          title={`${LABELS[key]}: ${summary[key]}`}
        />
      ))}
    </div>
  )
}

export function ProvenanceLegend() {
  return (
    <div className="flex flex-wrap gap-x-3 gap-y-1 text-[10px] text-muted-foreground">
      {Object.entries(PROVENANCE_COLORS).map(([key, color]) => (
        <span key={key} className="flex items-center gap-1">
          <span className={cn('inline-block size-2 rounded-sm', color)} />
          {LABELS[key]}
        </span>
      ))}
    </div>
  )
}
