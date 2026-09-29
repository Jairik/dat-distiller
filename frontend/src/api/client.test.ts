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

  it('turns a 422 validation detail array into a readable message', async () => {
    // The `msgs` are the field-level reasons FastAPI goes to real trouble to
    // produce. They were captured onto `detail` and thrown away, so a Balance
    // Target above 1 or an unknown column read as "status 422" and nothing more.
    const detail = [
      {
        loc: ['body', 'balance', 'topic', 'positive'],
        msg: 'Input should be less than or equal to 1',
        type: 'less_than_equal',
      },
    ]
    stubFetch(response(422, { detail }, 'Unprocessable Entity'))
    const error = await expectApiError(apiPost('/generate', { rows: 'many' }))
    expect(error.status).toBe(422)
    // The structure is still there for a caller that wants it.
    expect(error.detail).toEqual(detail)
    expect(error.message).toBe('balance.topic.positive: Input should be less than or equal to 1')
  })

  it('joins several validation failures into one sentence', async () => {
    const detail = [
      { loc: ['body', 'count'], msg: 'Input should be greater than 0', type: 'greater_than' },
      { loc: ['body', 'mode'], msg: 'Input should be a valid string', type: 'string_type' },
    ]
    stubFetch(response(422, { detail }, 'Unprocessable Entity'))
    const error = await expectApiError(apiPost('/generate', {}))
    expect(error.message).toBe(
      'count: Input should be greater than 0; mode: Input should be a valid string',
    )
  })

  it('still falls back to the status when a detail array is not the usual shape', async () => {
    // An array that is not FastAPI's `{loc, msg}` shape must not produce a
    // message of "[object Object]" or an empty string.
    stubFetch(response(422, { detail: ['something', 'unexpected'] }, 'Unprocessable Entity'))
    const error = await expectApiError(apiPost('/generate', {}))
    expect(error.message).toBe('Request failed with status 422')
    expect(error.detail).toEqual(['something', 'unexpected'])
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
