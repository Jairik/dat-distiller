import { afterEach, describe, expect, it, vi } from 'vitest'

import { ApiError, apiDelete, apiGet, apiPost } from '@/api/client'

/** Minimal Response stand-in: the client only reads these fields. */
function response(status: number, body: unknown, statusText = ''): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText,
    json: async () => body,
  } as unknown as Response
}

function nonJsonResponse(status = 500, statusText = 'Internal Server Error'): Response {
  return {
    ok: false,
    status,
    statusText,
    json: async () => {
      throw new TypeError('Unexpected end of JSON input')
    },
  } as unknown as Response
}

function stubFetch(resolved: Response) {
  const fetchMock = vi.fn().mockResolvedValue(resolved)
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

async function expectApiError(promise: Promise<unknown>): Promise<ApiError> {
  const settled = await promise.then(
    () => null,
    (error: unknown) => error,
  )
  expect(settled).toBeInstanceOf(ApiError)
  return settled as ApiError
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('apiGet', () => {
  it('requests under /api and parses JSON', async () => {
    const fetchMock = stubFetch(response(200, { status: 'ok', version: '0.1.0' }))
    await expect(apiGet<{ status: string }>('/health')).resolves.toEqual({
      status: 'ok',
      version: '0.1.0',
    })
    expect(fetchMock).toHaveBeenCalledWith('/api/health', undefined)
  })

  it('throws ApiError carrying the FastAPI detail string', async () => {
    stubFetch(response(404, { detail: 'Project not found' }, 'Not Found'))
    const error = await expectApiError(apiGet('/projects/7'))
    expect(error.status).toBe(404)
    expect(error.message).toBe('Project not found')
    expect(error.detail).toBe('Project not found')
  })

  it('keeps 422 validation detail arrays', async () => {
    const detail = [{ loc: ['body', 'rows'], msg: 'Input should be a valid integer' }]
    stubFetch(response(422, { detail }, 'Unprocessable Entity'))
    const error = await expectApiError(apiPost('/generate', { rows: 'many' }))
    expect(error.status).toBe(422)
    expect(error.detail).toEqual(detail)
    expect(error.message).toContain('422')
  })

  it('falls back to the status text for non-JSON bodies', async () => {
    stubFetch(nonJsonResponse())
    const error = await expectApiError(apiGet('/health'))
    expect(error.status).toBe(500)
    expect(error.detail).toBe('Internal Server Error')
    expect(error.message).toBe('Internal Server Error')
  })
})

describe('apiPost', () => {
  it('sends the body as JSON', async () => {
    const fetchMock = stubFetch(response(201, { id: 'p1' }))
    await expect(apiPost<{ id: string }>('/projects', { name: 'Demo' })).resolves.toEqual({
      id: 'p1',
    })
    const [path, init] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(path).toBe('/api/projects')
    expect(init.method).toBe('POST')
    expect(init.headers).toMatchObject({ 'Content-Type': 'application/json' })
    expect(JSON.parse(String(init.body))).toEqual({ name: 'Demo' })
  })
})

describe('apiDelete', () => {
  it('uses DELETE and tolerates an empty 204 response', async () => {
    const fetchMock = stubFetch(response(204, undefined, 'No Content'))
    await expect(apiDelete('/projects/1')).resolves.toBeUndefined()
    const [path, init] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(path).toBe('/api/projects/1')
    expect(init.method).toBe('DELETE')
  })
})
