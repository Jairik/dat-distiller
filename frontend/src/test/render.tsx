/**
 * Test render harness: fresh QueryClient (no retries) + MemoryRouter at a
 * chosen route, mirroring the real app's providers.
 */

import { render } from '@testing-library/react'
import { vi } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router-dom'
import type { ReactElement } from 'react'

export function renderWithProviders(ui: ReactElement, { route = '/' } = {}) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[route]}>{ui}</MemoryRouter>
    </QueryClientProvider>,
  )
}

/** fetch stub: routes by `METHOD path` pattern → JSON factory. */
export function mockFetch(
  handlers: Record<string, () => unknown | Promise<unknown>>,
): { calls: Array<[string, string]> } {
  const calls: Array<[string, string]> = []
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const method = (init?.method ?? 'GET').toUpperCase()
      const full = String(input).replace(/^\/api/, '') || '/'
      const path = full.split('?')[0]
      calls.push([method, full])
      const handler =
        handlers[`${method} ${full}`] ?? handlers[full] ?? handlers[`${method} ${path}`] ?? handlers[path]
      if (!handler) {
        return json({ detail: `unmocked ${method} ${path}` }, 404)
      }
      const body = await handler()
      return json(body, 200)
    }),
  )
  return { calls }
}

export function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}
