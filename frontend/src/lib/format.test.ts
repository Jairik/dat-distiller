import { describe, expect, it } from 'vitest'

import { fmtBytes, fmtDateTime, fmtNumber, timeAgo } from '@/lib/format'

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
