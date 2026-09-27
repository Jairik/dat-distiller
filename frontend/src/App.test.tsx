import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import App from '@/App'

function renderApp() {
  return render(
    <QueryClientProvider client={new QueryClient()}>
      <App />
    </QueryClientProvider>,
  )
}

describe('App', () => {
  it('renders the Dat Distiller title', () => {
    renderApp()
    expect(screen.getByRole('heading', { level: 1, name: 'Dat Distiller' })).toBeInTheDocument()
  })

  it('lists the three steps inside the staggered entrance', () => {
    renderApp()
    for (const step of ['Generate', 'Label', 'Train']) {
      expect(screen.getByRole('button', { name: step })).toBeInTheDocument()
    }
  })
})
