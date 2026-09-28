/**
 * Settings page: the four sections and the promises they make.
 *
 * The two rules worth protecting are that a key value never comes back to the
 * browser (so a saved key is masked and the field is emptied) and that an
 * environment-sourced key cannot be edited here at all, because the backend
 * would ignore the write anyway.
 */

import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import App from '@/App'
import { json, mockFetch, renderWithProviders } from '@/test/render'

/** The stubbed `navigator.clipboard.writeText`, so a test can assert on it. */
let writeText: ReturnType<typeof vi.fn>

/**
 * `userEvent.setup()` installs its own `navigator.clipboard`, so this has to run
 * after it — otherwise the assertion would be about user-event's stub, not ours.
 */
function spyOnClipboard() {
  writeText = vi.fn(async () => undefined)
  Object.defineProperty(navigator, 'clipboard', {
    value: { writeText },
    configurable: true,
  })
}

afterEach(() => {
  vi.unstubAllGlobals()
})

const settings = {
  default_provider: 'claude',
  models: { openrouter: 'anthropic/claude-sonnet-4.5' },
  soft_limits: { upload_rows: 10000, generation_rows: 5000, labeling_calls: 200 },
  review_threshold: 0.8,
  fairness_gap_threshold: 0.1,
  keys: {
    openrouter: { set: true, source: 'file' },
    typesafe: { set: false, source: null },
  },
  providers: [
    { id: 'claude', kind: 'cli', available: true },
    { id: 'codex', kind: 'cli', available: false },
    { id: 'opencode', kind: 'cli', available: false },
    { id: 'openrouter', kind: 'http', available: true },
  ],
}

const health = {
  status: 'ok',
  version: '0.1.0',
  extras: { torch: false, tensorflow: false, presidio: true },
}

const openrouterModels = {
  models: [
    { id: 'anthropic/claude-sonnet-4.5', name: 'Claude Sonnet 4.5' },
    { id: 'openai/gpt-5', name: 'GPT-5' },
  ],
}

const baseHandlers = {
  'GET /settings': () => settings,
  'GET /health': () => health,
  'GET /providers/openrouter/models': () => openrouterModels,
}

/**
 * Keys and Providers each render one row per entry, in a fixed order (the API
 * sends them: `openrouter` then `typesafe`; `claude`, `codex`, `opencode`,
 * `openrouter`). Selectors below index into that order rather than reaching for
 * copy that is not unique to one row.
 */
const KEY_ORDER = ['openrouter', 'typesafe'] as const
const PROVIDER_ORDER = ['claude', 'codex', 'opencode', 'openrouter'] as const

function saveKeyButton(index: number) {
  return screen.getAllByRole('button', { name: 'Save key' })[index]
}

function modelSelect(provider: (typeof PROVIDER_ORDER)[number]) {
  return document.getElementById(`model-${provider}`) as HTMLSelectElement
}

function renderSettings() {
  return renderWithProviders(<App />, { route: '/settings' })
}

describe('Settings — API keys', () => {
  it('masks a stored key, badges its source, and never shows the value', async () => {
    const { calls } = mockFetch(baseHandlers)
    renderSettings()

    expect(await screen.findByText('••••••••••••')).toBeInTheDocument()
    expect(screen.getByText('set (file)')).toBeInTheDocument()
    expect(screen.getByText('not set')).toBeInTheDocument()
    // nothing the API sent contains a key value, and nothing rendered echoes one
    expect(calls.some(([, p]) => p.includes('sk-'))).toBe(false)
    expect(screen.getByLabelText('New OpenRouter key')).toHaveAttribute('type', 'password')
  })

  it('saves a typed key once, then empties the field', async () => {
    const user = userEvent.setup()
    const { calls } = mockFetch({
      ...baseHandlers,
      'PUT /settings/keys/openrouter': () => ({ keys: settings.keys }),
    })
    renderSettings()
    const field = await screen.findByLabelText('New OpenRouter key')

    await user.type(field, 'sk-or-v1-secret')
    await user.click(saveKeyButton(KEY_ORDER.indexOf('openrouter')))

    await waitFor(() => expect(calls).toContainEqual(['PUT', '/settings/keys/openrouter']))
    // write-only: the value is sent, then dropped from the DOM
    await waitFor(() => expect(field).toHaveValue(''))
    expect(await screen.findByText('OpenRouter key saved')).toBeInTheDocument()
  })

  it('clears a stored key', async () => {
    const user = userEvent.setup()
    const { calls } = mockFetch({
      ...baseHandlers,
      'DELETE /settings/keys/openrouter': () => ({ keys: settings.keys }),
    })
    renderSettings()
    await screen.findByText('set (file)')

    await user.click(screen.getByRole('button', { name: 'Clear' }))
    await waitFor(() => expect(calls).toContainEqual(['DELETE', '/settings/keys/openrouter']))
  })

  it('shows an env-sourced key as read-only and explains why', async () => {
    const user = userEvent.setup()
    const env = {
      ...settings,
      keys: { openrouter: { set: true, source: 'env' }, typesafe: { set: false, source: null } },
    }
    const { calls } = mockFetch({ ...baseHandlers, 'GET /settings': () => env })
    renderSettings()

    expect(await screen.findByText('set (env)')).toBeInTheDocument()
    expect(screen.getByText(/always prefers over a stored key/)).toBeInTheDocument()
    expect(screen.getByText('OPENROUTER_API_KEY')).toBeInTheDocument()
    // no field and no Clear: the write would be ignored, so do not offer it
    expect(screen.queryByLabelText('New OpenRouter key')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Clear' })).not.toBeInTheDocument()
    expect(calls.some(([method]) => method === 'PUT')).toBe(false)
    expect(user).toBeTruthy()
  })

  it('reports a key the backend refused', async () => {
    const user = userEvent.setup()
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const path = String(input).replace(/^\/api/, '')
        if (init?.method === 'PUT') {
          return new Response(JSON.stringify({ detail: 'unknown provider' }), {
            status: 422,
            headers: { 'Content-Type': 'application/json' },
          })
        }
        const table: Record<string, unknown> = { '/settings': settings, '/health': health }
        return new Response(JSON.stringify(table[path]), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        })
      }),
    )
    renderSettings()
    await user.type(await screen.findByLabelText('New OpenRouter key'), 'nope')
    await user.click(saveKeyButton(KEY_ORDER.indexOf('openrouter')))
    expect(await screen.findByRole('alert')).toHaveTextContent('unknown provider')
  })
})

describe('Settings — Providers', () => {
  it('badges each Provider and names what is missing', async () => {
    mockFetch(baseHandlers)
    renderSettings()

    // the default-Provider dropdown and the per-Provider rows both name them
    await screen.findAllByText('Claude Code')
    expect(screen.getAllByText('Codex').length).toBeGreaterThan(0)
    // one CLI and OpenRouter are ready; two CLIs are not — the badge says which
    expect(screen.getAllByText('available')).toHaveLength(2)
    expect(screen.getAllByText('not installed')).toHaveLength(2)
    expect(screen.getAllByText(/Not on PATH/)).toHaveLength(2)
    // an unavailable http Provider is missing a key, not an install
    const noKey = {
      ...settings,
      keys: { openrouter: { set: false, source: null }, typesafe: { set: false, source: null } },
      providers: settings.providers.map((p) =>
        p.id === 'openrouter' ? { ...p, available: false } : p,
      ),
    }
    mockFetch({ ...baseHandlers, 'GET /settings': () => noKey })
    const second = renderSettings()
    expect(await screen.findByText('no key set')).toBeInTheDocument()
    expect(second.container).toHaveTextContent('Set the OpenRouter key above')
  })

  it('offers a copyable install command for each missing CLI', async () => {
    const user = userEvent.setup()
    spyOnClipboard()
    mockFetch(baseHandlers)
    renderSettings()
    await screen.findAllByText('Codex')

    expect(screen.getByText('npm i -g @openai/codex')).toBeInTheDocument()
    expect(screen.getByText('npm i -g opencode-ai')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Copy Codex install command' }))
    await waitFor(() =>
      expect(writeText).toHaveBeenCalledWith('npm i -g @openai/codex'),
    )
    expect(await screen.findByText('Copied')).toBeInTheDocument()
  })

  it('saves the default Provider and confirms it', async () => {
    const user = userEvent.setup()
    const { calls } = mockFetch({
      ...baseHandlers,
      'PUT /settings': () => ({ ...settings, default_provider: 'codex' }),
    })
    renderSettings()
    const select = await screen.findByLabelText('Default Provider')
    expect(select).toHaveValue('claude')

    await user.selectOptions(select, 'codex')
    await waitFor(() =>
      expect(
        calls.some(
          ([method, p]) => method === 'PUT' && p === '/settings',
        ),
      ).toBe(true),
    )
    expect(await screen.findByText('Default Provider saved')).toBeInTheDocument()
  })

  it('offers the OpenRouter catalogue once a key is set, plus agent default', async () => {
    const user = userEvent.setup()
    const { calls } = mockFetch({
      ...baseHandlers,
      'PUT /settings': () => settings,
    })
    renderSettings()
    const select = await screen.findByLabelText('Model', {
      selector: '#model-openrouter',
    })
    // the catalogue shows here and only here: a CLI Provider cannot call these
    await waitFor(() =>
      expect(within(select).getByRole('option', { name: 'GPT-5' })).toBeInTheDocument(),
    )
    expect(within(modelSelect('claude')).queryByRole('option', { name: 'GPT-5' })).not.toBeInTheDocument()
    expect(select).toHaveValue('anthropic/claude-sonnet-4.5')
    // the empty id is the "agent default" option
    expect(within(select).getByRole('option', { name: 'agent default' })).toBeInTheDocument()

    await user.selectOptions(select, 'openai/gpt-5')
    await waitFor(() => expect(calls.some(([m, p]) => m === 'PUT' && p === '/settings')).toBe(true))

    await user.selectOptions(select, '')
    expect(await screen.findByText(/uses agent default/)).toBeInTheDocument()
  })

  it('falls back to a free-text model id for a CLI Provider', async () => {
    // a CLI Provider has no catalogue to list, so its model id is typed
    mockFetch(baseHandlers)
    renderSettings()
    await screen.findAllByText('Codex')

    const claude = modelSelect('claude')
    expect(claude).toBeInTheDocument()
    // no OpenRouter options leaked into a CLI Provider's list
    expect(within(claude).queryByRole('option', { name: 'GPT-5' })).not.toBeInTheDocument()
    // the field is already open, because there is nothing to pick from
    expect(document.getElementById('model-id-claude')).toBeInTheDocument()
  })

  it('never asks OpenRouter for a catalogue without a key', async () => {
    const { calls } = mockFetch({
      ...baseHandlers,
      'GET /settings': () => ({
        ...settings,
        keys: { openrouter: { set: false, source: null }, typesafe: { set: false, source: null } },
        providers: settings.providers.map((p) =>
          p.id === 'openrouter' ? { ...p, available: false } : p,
        ),
      }),
    })
    renderSettings()
    await screen.findAllByText('not set')
    expect(calls.some(([, p]) => p.startsWith('/providers/openrouter/models'))).toBe(false)
  })
})

describe('Settings — limits and thresholds', () => {
  it('saves soft limits and thresholds together, with confirmation', async () => {
    const user = userEvent.setup()
    let sent: Record<string, unknown> | null = null
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const path = String(input).replace(/^\/api/, '')
        if (init?.method === 'PUT' && path === '/settings') {
          sent = JSON.parse(String(init.body))
          return json(settings)
        }
        const table: Record<string, unknown> = {
          '/settings': settings,
          '/health': health,
        }
        if (path.startsWith('/providers/openrouter')) return json(openrouterModels)
        return json(table[path])
      }),
    )
    renderSettings()
    const upload = await screen.findByLabelText('Upload rows')
    expect(upload).toHaveValue(10000)

    await user.clear(upload)
    await user.type(upload, '20000')
    await user.click(screen.getByRole('button', { name: 'Save limits' }))

    await waitFor(() => expect(sent).not.toBeNull())
    expect(sent).toEqual({
      soft_limits: { upload_rows: 20000, generation_rows: 5000, labeling_calls: 200 },
      review_threshold: 0.8,
      fairness_gap_threshold: 0.1,
    })
    expect(await screen.findByText('Limits saved')).toBeInTheDocument()
  })

  it('refuses a nonsense limit before sending it', async () => {
    const user = userEvent.setup()
    const { calls } = mockFetch(baseHandlers)
    renderSettings()
    const generation = await screen.findByLabelText('Generation rows')

    await user.clear(generation)
    await user.type(generation, '0')
    await user.click(screen.getByRole('button', { name: 'Save limits' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('at least 1')
    expect(calls.some(([m]) => m === 'PUT')).toBe(false)
  })

  it('refuses a threshold outside 0..1', async () => {
    const user = userEvent.setup()
    const { calls } = mockFetch(baseHandlers)
    renderSettings()
    const review = await screen.findByLabelText('Review threshold')

    await user.clear(review)
    await user.type(review, '1.5')
    await user.click(screen.getByRole('button', { name: 'Save limits' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('between 0 and 1')
    expect(calls.some(([m]) => m === 'PUT')).toBe(false)
  })
})

describe('Settings — optional extras', () => {
  it('lists each extra with its status and an install command when missing', async () => {
    mockFetch(baseHandlers)
    renderSettings()

    expect(await screen.findByText('Torch')).toBeInTheDocument()
    expect(screen.getByText('installed')).toBeInTheDocument() // presidio
    expect(screen.getAllByText('missing')).toHaveLength(2)
    expect(screen.getByText('uv sync --extra torch')).toBeInTheDocument()
    expect(screen.getByText('uv sync --extra tensorflow')).toBeInTheDocument()
    // an installed extra needs no install hint
    expect(screen.queryByText('uv sync --extra presidio')).not.toBeInTheDocument()
  })

  it('stays calm when the health endpoint is unavailable', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const path = String(input).replace(/^\/api/, '')
        if (path === '/health') return new Response('', { status: 503 })
        return new Response(JSON.stringify(settings), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        })
      }),
    )
    renderSettings()
    expect(await screen.findByText(/Could not read install status/)).toBeInTheDocument()
  })
})
