import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { HoverPress } from '@/components/motion'

describe('HoverPress', () => {
  it('renders children and passes clicks through', async () => {
    const onClick = vi.fn()
    const user = userEvent.setup()
    render(
      <HoverPress>
        <button type="button" onClick={onClick}>
          Start a Project
        </button>
      </HoverPress>,
    )
    await user.click(screen.getByRole('button', { name: 'Start a Project' }))
    expect(onClick).toHaveBeenCalledTimes(1)
  })
})
