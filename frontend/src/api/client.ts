/**
 * Typed client for the Dat Distiller API.
 *
 * Every path is relative to `/api` (proxied to the backend in dev, same-origin
 * under `dat-distiller serve`). FastAPI errors arrive as `{"detail": ...}` and
 * are rethrown as `ApiError` so callers can branch on `status`/`detail`.
 */

export class ApiError extends Error {
  /** HTTP status code of the failed response. */
  readonly status: number
  /**
   * The `detail` value from the FastAPI error body: usually a string, or an
   * array of validation errors for 422s. Falls back to the status text when
   * the body was not JSON.
   */
  readonly detail: unknown

  constructor(status: number, detail: unknown, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

const API_BASE = '/api'

/**
 * FastAPI's validation errors, as one sentence.
 *
 * A 422 body is a list of `{loc, msg, type}`. The `msg`s are the field-level
 * reasons FastAPI goes to real trouble to produce, and they were being captured
 * onto `ApiError.detail` and then thrown away — so every validation failure
 * reached the person as "Request failed with status 422". The full structure
 * stays on `detail` for callers that want it; this is only the message.
 */
function describeValidationDetail(detail: unknown): string | null {
  if (!Array.isArray(detail) || detail.length === 0) return null
  const parts: string[] = []
  for (const entry of detail) {
    if (entry === null || typeof entry !== 'object') return null
    const { loc, msg } = entry as { loc?: unknown; msg?: unknown }
    if (typeof msg !== 'string' || msg.length === 0) return null
    // `loc` starts with ('body'|'query'|'path'), which says nothing to a reader.
    const field = Array.isArray(loc) ? loc.slice(1).map(String).join('.') : ''
    parts.push(field ? `${field}: ${msg}` : msg)
  }
  return parts.join('; ') || null
}

async function toApiError(response: Response): Promise<ApiError> {
  let detail: unknown = response.statusText || `HTTP ${response.status}`
  try {
    const body: unknown = await response.json()
    if (body !== null && typeof body === 'object' && 'detail' in body) {
      detail = (body as { detail: unknown }).detail
    } else {
      detail = body
    }
  } catch {
    // Not JSON (a proxy error page, an empty body): keep the status text.
  }
  const message =
    typeof detail === 'string' && detail.length > 0
      ? detail
      : (describeValidationDetail(detail) ??
        `Request failed with status ${response.status}`)
  return new ApiError(response.status, detail, message)
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init)
  if (!response.ok) {
    throw await toApiError(response)
  }
  if (response.status === 204) {
    return undefined as T
  }
  return (await response.json()) as T
}

/** GET `path` and parse the JSON response. */
export function apiGet<T>(path: string): Promise<T> {
  return request<T>(path)
}

/**
 * POST `body` to `path` and parse the JSON response.
 *
 * A `FormData` body is passed through **untouched** and gets no explicit
 * `Content-Type`: the browser has to set that itself so it can include the
 * multipart boundary. Stringifying it would send the literal text
 * `"[object FormData]"` and the server would answer 422 for a reason that has
 * nothing to do with the data — which is exactly what happened to the
 * predict-on-CSV upload before the end-to-end suite found it.
 */
export function apiPost<T>(path: string, body: unknown): Promise<T> {
  if (body instanceof FormData) {
    return request<T>(path, { method: 'POST', body })
  }
  return request<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

/** PUT `body` as JSON to `path` and parse the JSON response. */
export function apiPut<T>(path: string, body: unknown): Promise<T> {
  if (body instanceof FormData) {
    return request<T>(path, { method: 'PUT', body })
  }
  return request<T>(path, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

/** DELETE `path` (204 responses carry no body). */
export function apiDelete(path: string): Promise<void> {
  return request<void>(path, { method: 'DELETE' })
}

/** POST a single file as multipart/form-data (field name: `file`). */
export function apiUpload<T>(path: string, file: File): Promise<T> {
  const form = new FormData()
  form.append('file', file)
  return request<T>(path, { method: 'POST', body: form })
}
