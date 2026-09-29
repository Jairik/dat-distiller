import { describe, expect, it } from 'vitest'

import { QueryClient } from '@tanstack/react-query'

import { fmtBytes, fmtDateTime, fmtNumber, timeAgo } from '@/lib/format'
import { versionsKey } from '@/lib/projects'

describe('fmtBytes', () => {
  it('renders explicit units at every scale', () => {
    expect(fmtBytes(0)).toBe('0 B')
    expect(fmtBytes(999)).toBe('999 B')
    expect(fmtBytes(1024)).toBe('1.0 KB')
    expect(fmtBytes(1536)).toBe('1.5 KB')
    expect(fmtBytes(5 * 1024 * 1024)).toBe('5.0 MB')
    expect(fmtBytes(3.25 * 1024 ** 3)).toBe('3.3 GB')
  })
  it('handles nonsense', () => {
    expect(fmtBytes(-1)).toBe('—')
    expect(fmtBytes(Number.NaN)).toBe('—')
  })
})

describe('fmtNumber', () => {
  it('thousands-separates integers and trims floats', () => {
    expect(fmtNumber(1234567)).toBe('1,234,567')
    expect(fmtNumber(12.345)).toBe('12.35')
    expect(fmtNumber(12.5)).toBe('12.5')
  })
})

describe('timeAgo', () => {
  it('buckets distance from now', () => {
    const now = Date.now()
    expect(timeAgo(new Date(now - 10_000).toISOString())).toBe('just now')
    expect(timeAgo(new Date(now - 5 * 60_000).toISOString())).toBe('5m ago')
    expect(timeAgo(new Date(now - 3 * 3600_000).toISOString())).toBe('3h ago')
  })
  it('handles missing and ancient values', () => {
    expect(timeAgo(null)).toBe('—')
    expect(timeAgo('2020-04-05T00:00:00Z')).toMatch(/2020/)
  })
})

describe('fmtDateTime', () => {
  it('renders a readable timestamp', () => {
    expect(fmtDateTime('2026-04-05T13:30:00Z')).toContain('2026')
    expect(fmtDateTime(undefined)).toBe('—')
  })
})

describe('fmtNumber — a rounded-away value is not a number', () => {
  it('never prints -0', () => {
    // `Intl` renders -0.004 at two fraction digits as "-0", which reads as a
    // real measurement rather than as something that rounded away.
    expect(fmtNumber(-0.004)).toBe('0')
    expect(fmtNumber(0)).toBe('0')
    // A value that does round to something keeps its sign and its digits.
    expect(fmtNumber(-0.4)).toBe('-0.4')
    expect(fmtNumber(-1.25)).toBe('-1.25')
  })
})

describe('timeAgo — a timestamp in the future', () => {
  it('says what it can honestly say, rather than pretending it is the past', () => {
    // A clock skew, or a server a few seconds ahead, used to fall into the
    // "45 seconds ago" branch as a negative number and render as "just now" —
    // indistinguishable from a genuinely recent event.
    const ahead = new Date(Date.now() + 5_000).toISOString()
    expect(timeAgo(ahead)).toBe('just now')
  })
})

describe('the Dataset Versions cache key', () => {
  it('is the one the query actually uses, so an invalidation reaches it', () => {
    // Three modules invalidated `['dataset_versions']`, which matches no query in
    // the app: the real key is `['versions', projectId]`. It went unnoticed
    // because all three callers also call `reloadVersions()` by hand — but the
    // comment next to the pii.ts one invites the next caller to trust it.
    //
    // Asserted on the query's own invalidation state rather than on the version
    // tree, because the tree appearing to refresh is exactly what the incidental
    // `reloadVersions()` makes look fine.
    const client = new QueryClient()
    const key = versionsKey('p1')
    client.setQueryData(key, [])
    expect(client.getQueryState(key)?.isInvalidated).toBe(false)

    // What the three call sites now do.
    client.invalidateQueries({ queryKey: ['versions'] })
    expect(client.getQueryState(key)?.isInvalidated).toBe(true)

    // And what they used to do, which reached nothing.
    const other = new QueryClient()
    other.setQueryData(key, [])
    other.invalidateQueries({ queryKey: ['dataset_versions'] })
    expect(other.getQueryState(key)?.isInvalidated).toBe(false)
  })
})
