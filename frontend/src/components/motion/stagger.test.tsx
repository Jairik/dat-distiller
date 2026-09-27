import { render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { StaggerGroup, StaggerItem } from '@/components/motion'
import { setPrefersReducedMotion } from '@/test/reduced-motion'

const STEPS = ['Generate', 'Label', 'Train']

function StepList() {
  return (
    <StaggerGroup>
      {STEPS.map((step) => (
        <StaggerItem key={step}>
          <span>{step}</span>
        </StaggerItem>
      ))}
    </StaggerGroup>
  )
}

afterEach(() => {
  setPrefersReducedMotion(false)
})

describe('StaggerGroup', () => {
  it('renders every StaggerItem child', () => {
    render(<StepList />)
    for (const step of STEPS) {
      expect(screen.getByText(step)).toBeInTheDocument()
    }
  })

  it('renders every child under prefers-reduced-motion too', () => {
    setPrefersReducedMotion(true)
    render(<StepList />)
    for (const step of STEPS) {
      expect(screen.getByText(step)).toBeInTheDocument()
    }
  })
})
