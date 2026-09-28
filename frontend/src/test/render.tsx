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

/**
 * fetch stub: routes by `METHOD path` pattern → a factory.
 *
 * Records both the requests made (`calls`) and the JSON bodies sent
 * (`bodies`), because "did the UI send the right thing" is as much a part of a
 * feature as "did it show the right thing". `lastBody` is the convenience for
 * the common single-POST case.
 */
export function mockFetch(
  handlers: Record<string, () => unknown | Promise<unknown>>,
): { calls: Array<[string, string]>; bodies: unknown[]; lastBody: () => unknown } {
  const calls: Array<[string, string]> = []
  const bodies: unknown[] = []
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const method = (init?.method ?? 'GET').toUpperCase()
      const full = String(input).replace(/^\/api/, '') || '/'
      const path = full.split('?')[0]
      calls.push([method, full])
      if (typeof init?.body === 'string') {
        try {
          bodies.push(JSON.parse(init.body))
        } catch {
          bodies.push(init.body)
        }
      }
      const handler =
        handlers[`${method} ${full}`] ?? handlers[full] ?? handlers[`${method} ${path}`] ?? handlers[path]
      if (!handler) {
        return json({ detail: `unmocked ${method} ${path}` }, 404)
      }
      const body = await handler()
      // A handler may return a Response of its own for endpoints that serve
      // something other than JSON (a Markdown Card, a CSV download). Passing it
      // through beats encoding a string and having the test decode it.
      if (body instanceof Response) return body
      return json(body, 200)
    }),
  )
  return { calls, bodies, lastBody: () => bodies.at(-1) }
}

export function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

/** A raw text response, for the endpoints that serve Markdown or CSV. */
export function text(body: string, status = 200, contentType = 'text/plain'): Response {
  return new Response(body, { status, headers: { 'Content-Type': contentType } })
}
