import { render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { AnimatedNumber } from '@/components/motion'
import { setPrefersReducedMotion } from '@/test/reduced-motion'

afterEach(() => {
  setPrefersReducedMotion(false)
})

describe('AnimatedNumber', () => {
  it('shows the final value immediately under prefers-reduced-motion', () => {
    setPrefersReducedMotion(true)
    render(<AnimatedNumber value={42} />)
    // No counting: the very first render must already be the target value.
    expect(screen.getByText('42')).toBeInTheDocument()
  })

  it('counts up to the final value with normal motion', async () => {
    render(<AnimatedNumber value={42} duration={0.05} />)
    // Starts below the target and lands exactly on it once the animation ends.
    await screen.findByText('42', undefined, { timeout: 2000 })
  })

  it('applies a custom format function', () => {
    setPrefersReducedMotion(true)
    render(<AnimatedNumber value={0.75} format={(value) => `${Math.round(value * 100)}%`} />)
    expect(screen.getByText('75%')).toBeInTheDocument()
  })
})
