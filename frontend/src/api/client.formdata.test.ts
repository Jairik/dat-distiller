/**
 * The request bodies the client actually puts on the wire.
 *
 * The FormData case exists because it was **broken** and no unit test caught it:
 * every earlier test stubbed `fetch`, so a body that was stringified into
 * `"[object FormData]"` was never seen by anything that could object. Asserting
 * on the `RequestInit` the client hands to `fetch` is the only place that bug is
 * visible without a server.
 */

import { afterEach, describe, expect, it, vi } from 'vitest'

import { apiPost, apiPut } from '@/api/client'

afterEach(() => {
  vi.unstubAllGlobals()
})

function captureFetch() {
  const calls: Array<[string, RequestInit | undefined]> = []
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      calls.push([String(input), init])
      return new Response('{}', { status: 200, headers: { 'Content-Type': 'application/json' } })
    }),
  )
  return calls
}

describe('apiPost with a FormData body', () => {
  it('passes the form through, not its stringification', async () => {
    const calls = captureFetch()
    const form = new FormData()
    form.append('file', new File(['a,b\n1,2\n'], 'new.csv', { type: 'text/csv' }))

    await apiPost('/train/runs/j1/predict?model=m', form)

    const [path, init] = calls[0]
    expect(path).toBe('/api/train/runs/j1/predict?model=m')
    expect(init?.body).toBe(form)
    expect(init?.body).toBeInstanceOf(FormData)
    expect((init?.body as FormData).get('file')).toBeInstanceOf(File)
  })

  it('sets no Content-Type, so the browser can add the multipart boundary', async () => {
    const calls = captureFetch()
    const form = new FormData()
    form.append('file', new File(['x'], 'x.csv', { type: 'text/csv' }))

    await apiPost('/predict', form)

    // a hand-set application/json here is the other half of the bug: the server
    // would be told the body is JSON when it is multipart
    const headers = (calls[0][1]?.headers ?? {}) as Record<string, string>
    expect(headers['Content-Type']).toBeUndefined()
  })

  it('still sends a plain object as JSON', async () => {
    const calls = captureFetch()
    await apiPost('/projects', { name: 'a project' })
    const [, init] = calls[0]
    expect(init?.body).toBe('{"name":"a project"}')
    expect((init?.headers as Record<string, string>)['Content-Type']).toBe('application/json')
  })

  it('handles a FormData on PUT the same way', async () => {
    const calls = captureFetch()
    const form = new FormData()
    form.append('file', new File(['y'], 'y.csv', { type: 'text/csv' }))
    await apiPut('/upload', form)
    expect(calls[0][1]?.body).toBe(form)
    expect(calls[0][1]?.method).toBe('PUT')
  })
})
