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
      : `Request failed with status ${response.status}`
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

/** POST `body` as JSON to `path` and parse the JSON response. */
export function apiPost<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

/** DELETE `path` (204 responses carry no body). */
export function apiDelete(path: string): Promise<void> {
  return request<void>(path, { method: 'DELETE' })
}
