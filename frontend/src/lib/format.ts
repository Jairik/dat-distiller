/**
 * Shared formatting helpers (bytes, dates, relative time).
 *
 * Bytes always render with explicit units (never a bare "1.5k") so numbers
 * read the same across the app.
 */

const KB = 1024

export function fmtBytes(n: number): string {
  if (!Number.isFinite(n) || n < 0) return '—'
  if (n < KB) return `${Math.round(n)} B`
  const units = ['KB', 'MB', 'GB', 'TB']
  let value = n / KB
  let i = 0
  while (value >= KB && i < units.length - 1) {
    value = value / KB
    i += 1
  }
  return `${value.toFixed(value < 10 ? 1 : 0)} ${units[i]}`
}

export function fmtNumber(n: number): string {
  if (!Number.isFinite(n)) return '—'
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: n % 1 === 0 ? 0 : 2 }).format(n)
}

/** "just now", "5m ago", "3h ago", "Apr 5", or "Apr 5 2024" for far dates. */
export function timeAgo(iso: string | null | undefined): string {
  if (!iso) return '—'
  const then = new Date(iso).getTime()
  if (Number.isNaN(then)) return '—'
  const seconds = (Date.now() - then) / 1000
  if (seconds < 45) return 'just now'
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h ago`
  const date = new Date(then)
  const sameYear = date.getFullYear() === new Date().getFullYear()
  return date.toLocaleDateString('en-US', sameYear ? { month: 'short', day: 'numeric' } : { month: 'short', day: 'numeric', year: 'numeric' })
}

export function fmtDateTime(iso: string | null | undefined): string {
  if (!iso) return '—'
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return '—'
  return date.toLocaleString('en-US', { dateStyle: 'medium', timeStyle: 'short' })
}
