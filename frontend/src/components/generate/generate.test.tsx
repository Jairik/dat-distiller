/**
 * The Generate step, end to end on both paths.
 *
 * The one rule the whole step exists to enforce: a Generation Run costs Provider
 * calls, so the run button does not unlock until a preview has been seen. Both
 * source paths are driven all the way to a Dataset Version, and the Checks panel
 * has to be acknowledged before Label is reachable.
 */

import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import App from '@/App'
import { localSpecErrors, toSpec, toWireSpec } from '@/lib/generate'
import { json, mockFetch, renderWithProviders } from '@/test/render'

class FakeEventSource {
  static instances: FakeEventSource[] = []
  closed = false
  listeners = new Map<string, (event: MessageEvent) => void>()
  constructor(public url: string) {
    FakeEventSource.instances.push(this)
  }
  addEventListener(name: string, fn: EventListener) {
    this.listeners.set(name, fn as (event: MessageEvent) => void)
  }
  close() {
    this.closed = true
  }
  onerror: ((event: unknown) => void) | null = null
  emit(name: string, data: unknown) {
    this.listeners.get(name)?.({ data: JSON.stringify(data) } as MessageEvent)
  }
}

beforeEach(() => {
  FakeEventSource.instances = []
  vi.stubGlobal('EventSource', FakeEventSource)
})

afterEach(() => {
  vi.unstubAllGlobals()
})

const project = {
  id: 'p1',
  name: 'Churn Lab',
  created_at: '2024-01-01T00:00:00Z',
  version_count: 1,
  latest_version_id: 'v1',
}

const V1 = {
  id: 'v1',
  project_id: 'p1',
  parent_id: null,
  number: 1,
  origin: 'uploaded',
  row_count: 40,
  columns: [
    { name: 'age', kind: 'integer' },
    { name: 'plan', kind: 'categorical' },
  ],
  provenance_summary: { uploaded: 40 },
  seed: null,
  meta: {},
  created_at: '2024-01-01T00:00:00Z',
}

const baseHandlers = {
  'GET /projects': () => [project],
  'GET /projects/p1': () => project,
  'GET /projects/p1/dataset_versions': () => [V1],
  'GET /providers/options': () => ({
    providers: [
      { id: 'claude', available: true, reason: null, model: null },
      { id: 'codex', available: false, reason: 'CLI not found on PATH', model: null },
    ],
  }),
  'GET /checks': () => ({ checks: [], unacknowledged_warnings: 0 }),
}

const PREVIEW = {
  seed: 4242,
  rows: [
    { age: 31, plan: 'pro' },
    { age: 45, plan: 'basic' },
    { age: 22, plan: 'pro' },
  ],
  provider_calls: 1,
}

const ESTIMATE = { rows: 500, estimated_provider_calls: 4, uses_provider: true }

const RUN_RESULT = {
  version_id: 'v2',
  rows: 500,
  seed: 4242,
  provider_calls: 4,
  dropped_rows: 0,
  fidelity_warnings: [],
  checks_raised: ['c1'],
  pii: { columns: [], total_findings: 0 },
}

function renderStep() {
  return renderWithProviders(<App />, { route: '/projects/p1/generate' })
}

async function describeIt(user: ReturnType<typeof userEvent.setup>) {
  await user.type(await screen.findByLabelText(/What should this dataset contain/), 'support tickets')
}

describe('Generate step — the preview gate', () => {
  it('will not run anything before a preview, and says why', async () => {
    const user = userEvent.setup()
    const { calls } = mockFetch(baseHandlers)
    renderStep()

    const previewButton = await screen.findByRole('button', { name: /Preview 5 rows/ })
    expect(previewButton).toBeDisabled()
    expect(screen.getByText('Describe the dataset first.')).toBeInTheDocument()

    await describeIt(user)
    // still nothing to preview: no sample chosen
    expect(previewButton).toBeDisabled()
    expect(screen.getByText('Pick a sample Dataset Version.')).toBeInTheDocument()

    await user.selectOptions(await screen.findByLabelText('Sample Dataset Version'), 'v1')
    expect(previewButton).toBeEnabled()
    // and the run path does not exist yet at all
    expect(screen.queryByRole('button', { name: /Generate the full dataset/ })).not.toBeInTheDocument()
    expect(screen.getByText('Preview a few rows to unlock the full run.')).toBeInTheDocument()
    expect(calls.some(([m]) => m === 'POST')).toBe(false)
  })

  it('previews, estimates, then unlocks the run', async () => {
    const user = userEvent.setup()
    const { calls } = mockFetch({
      ...baseHandlers,
      'POST /generate/preview': () => PREVIEW,
      'POST /generate/estimate': () => ESTIMATE,
      'POST /generate/run': () => ({ id: 'j1' }),
    })
    renderStep()
    await describeIt(user)
    await user.selectOptions(await screen.findByLabelText('Sample Dataset Version'), 'v1')
    await user.click(screen.getByRole('button', { name: /Preview 5 rows/ }))

    // the preview is on screen before anything is spendable
    const table = await screen.findByRole('table')
    expect(within(table).getByText('age')).toBeInTheDocument()
    expect(within(table).getByText('45')).toBeInTheDocument()
    expect(screen.getByText(/nothing has been saved yet/)).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /Estimate and continue/ }))
    expect(await screen.findByText(/4 Provider call\(s\)/)).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /Generate the full dataset/ }))
    await waitFor(() => expect(calls).toContainEqual(['POST', '/generate/run']))
  })

  it('a free statistical Mode is honest about costing nothing', async () => {
    const user = userEvent.setup()
    mockFetch({
      ...baseHandlers,
      'POST /generate/preview': () => ({ ...PREVIEW, provider_calls: null }),
      'POST /generate/estimate': () => ({ ...ESTIMATE, uses_provider: false, estimated_provider_calls: 0 }),
    })
    renderStep()
    await describeIt(user)
    await user.selectOptions(await screen.findByLabelText('Sample Dataset Version'), 'v1')
    await user.click(await screen.findByRole('radio', { name: /statistical/ }))
    await user.click(screen.getByRole('button', { name: /Preview 5 rows/ }))
    await screen.findByRole('table')
    await user.click(screen.getByRole('button', { name: /Estimate and continue/ }))
    expect(await screen.findByText(/needs no Provider calls at all/)).toBeInTheDocument()
  })

  it('editing the description invalidates the preview, because it no longer describes the run', async () => {
    const user = userEvent.setup()
    mockFetch({ ...baseHandlers, 'POST /generate/preview': () => PREVIEW })
    renderStep()
    await describeIt(user)
    await user.selectOptions(await screen.findByLabelText('Sample Dataset Version'), 'v1')
    await user.click(screen.getByRole('button', { name: /Preview 5 rows/ }))
    await screen.findByRole('table')

    await user.type(screen.getByLabelText(/What should this dataset contain/), ' for a bank')
    expect(screen.queryByRole('table')).not.toBeInTheDocument()
    expect(screen.getByText('Preview a few rows to unlock the full run.')).toBeInTheDocument()
  })

  it('surfaces a preview failure without unlocking the run', async () => {
    const user = userEvent.setup()
    mockFetch({
      ...baseHandlers,
      'POST /generate/preview': () => {
        throw new Error('no Provider configured')
      },
    })
    renderStep()
    await describeIt(user)
    await user.selectOptions(await screen.findByLabelText('Sample Dataset Version'), 'v1')
    await user.click(screen.getByRole('button', { name: /Preview 5 rows/ }))
    expect(await screen.findByRole('alert')).toHaveTextContent('no Provider configured')
    expect(screen.queryByRole('button', { name: /Generate the full dataset/ })).not.toBeInTheDocument()
  })
})

describe('Generate step — the Column Specs path', () => {
  it('goes from a description to a run with no sample at all', async () => {
    const user = userEvent.setup()
    const { calls } = mockFetch({
      ...baseHandlers,
      'POST /generate/suggest-columns': () => ({
        specs: [
          { name: 'topic', type: 'categorical', min: null, max: null, categories: ['billing', 'bug'], description: null },
          { name: 'sentiment', type: 'categorical', min: null, max: null, categories: ['pos', 'neg'], description: null },
        ],
        validation_errors: [],
      }),
      'POST /generate/preview': () => PREVIEW,
      'POST /generate/estimate': () => ESTIMATE,
      'POST /generate/run': () => ({ id: 'j1' }),
    })
    renderStep()
    await describeIt(user)
    await user.click(await screen.findByRole('radio', { name: /Define the columns/ }))

    // the suggestion arrives and becomes an editable table
    await user.click(screen.getByRole('button', { name: /Suggest with Provider/ }))
    const names = await screen.findAllByLabelText('Name')
    expect(names).toHaveLength(2)
    expect(names[0]).toHaveValue('topic')
    const categories = await screen.findAllByLabelText('Categories (comma separated)')
    expect(categories[0]).toHaveValue('billing, bug')

    await user.click(screen.getByRole('button', { name: /Preview 5 rows/ }))
    await screen.findByRole('table')
    const preview = calls.find(([m, p]) => m === 'POST' && p === '/generate/preview')
    expect(preview).toBeDefined()

    await user.click(screen.getByRole('button', { name: /Estimate and continue/ }))
    await user.click(await screen.findByRole('button', { name: /Generate the full dataset/ }))
    await waitFor(() => expect(calls).toContainEqual(['POST', '/generate/run']))
  })

  it('refuses to preview while the Column Specs are invalid', async () => {
    const user = userEvent.setup()
    const { calls } = mockFetch(baseHandlers)
    renderStep()
    await describeIt(user)
    await user.click(await screen.findByRole('radio', { name: /Define the columns/ }))

    const names = await screen.findAllByLabelText('Name')
    await user.clear(names[0])
    expect(await screen.findByRole('alert')).toHaveTextContent('name is required')
    expect(screen.getByRole('button', { name: /Preview 5 rows/ })).toBeDisabled()
    expect(calls.some(([m]) => m === 'POST')).toBe(false)
  })

  it('rejects a categorical column with no categories, and a min above its max', () => {
    expect(
      localSpecErrors([{ name: 'plan', type: 'categorical', min: null, max: null, categories: [], description: null }]),
    ).toEqual(['plan: a categorical column needs at least one category'])
    expect(
      localSpecErrors([{ name: 'age', type: 'integer', min: 90, max: 18, categories: null, description: null }]),
    ).toEqual(['age: min must be <= max'])
    expect(
      localSpecErrors([
        { name: 'a', type: 'number', min: null, max: null, categories: null, description: null },
        { name: 'a', type: 'number', min: null, max: null, categories: null, description: null },
      ]),
    ).toEqual(['a: duplicate name'])
  })

  it('adds and removes rows in the editable table', async () => {
    const user = userEvent.setup()
    mockFetch(baseHandlers)
    renderStep()
    await describeIt(user)
    await user.click(await screen.findByRole('radio', { name: /Define the columns/ }))
    expect(await screen.findAllByLabelText('Name')).toHaveLength(1)

    await user.click(screen.getByRole('button', { name: /Add column/ }))
    expect(await screen.findAllByLabelText('Name')).toHaveLength(2)
    await user.click(screen.getByRole('button', { name: 'Remove column_2' }))
    await waitFor(() => expect(screen.getAllByLabelText('Name')).toHaveLength(1))
  })

  it('shows only min/max for a numeric column and categories for a categorical one', async () => {
    const user = userEvent.setup()
    mockFetch(baseHandlers)
    renderStep()
    await describeIt(user)
    await user.click(await screen.findByRole('radio', { name: /Define the columns/ }))
    expect(await screen.findByLabelText('Min')).toBeInTheDocument()
    expect(screen.queryByLabelText('Categories (comma separated)')).not.toBeInTheDocument()

    await user.selectOptions(screen.getByLabelText('Type'), 'categorical')
    expect(await screen.findByLabelText('Categories (comma separated)')).toBeInTheDocument()
    expect(screen.queryByLabelText('Min')).not.toBeInTheDocument()
  })

  it('cannot suggest columns without a description to suggest from', async () => {
    mockFetch(baseHandlers)
    renderStep()
    await userEvent.setup()
    await userEvent.click(await screen.findByRole('radio', { name: /Define the columns/ }))
    expect(screen.getByRole('button', { name: /Suggest with Provider/ })).toBeDisabled()
  })

  it('reports a Provider that could not suggest anything', async () => {
    const user = userEvent.setup()
    mockFetch({
      ...baseHandlers,
      'POST /generate/suggest-columns': () => {
        throw new Error('OpenRouter needs an API key')
      },
    })
    renderStep()
    await describeIt(user)
    await user.click(await screen.findByRole('radio', { name: /Define the columns/ }))
    await user.click(screen.getByRole('button', { name: /Suggest with Provider/ }))
    expect(await screen.findByRole('alert')).toHaveTextContent('needs an API key')
  })
})

describe('Generate step — the run and the gate to Label', () => {
  async function runToCompletion(user: ReturnType<typeof userEvent.setup>, handlers: object) {
    mockFetch({
      ...baseHandlers,
      'POST /generate/preview': () => PREVIEW,
      'POST /generate/estimate': () => ESTIMATE,
      'POST /generate/run': () => ({ id: 'j1' }),
      ...handlers,
    })
    renderStep()
    await describeIt(user)
    await user.selectOptions(await screen.findByLabelText('Sample Dataset Version'), 'v1')
    await user.click(screen.getByRole('button', { name: /Preview 5 rows/ }))
    await screen.findByRole('table')
    await user.click(screen.getByRole('button', { name: /Estimate and continue/ }))
    await user.click(await screen.findByRole('button', { name: /Generate the full dataset/ }))
  }

  it('follows the run through SSE and shows the new Dataset Version', async () => {
    const user = userEvent.setup()
    await runToCompletion(user, { 'GET /jobs/j1': () => ({ id: 'j1', status: 'running', progress: { done: 1, total: 4 }, result: null }) })

    expect(await screen.findByText('Generation Run')).toBeInTheDocument()
    const es = FakeEventSource.instances.at(-1)!
    es.emit('progress', { status: 'running', progress: { done: 4, total: 4 }, result: null, error: null })
    es.emit('completed', {
      status: 'completed',
      progress: { done: 4, total: 4 },
      result: RUN_RESULT,
      error: null,
    })

    const run = (await screen.findByTestId('job-j1')).textContent ?? ''
    expect(run).toMatch(/500 rows/)
    expect(run).toMatch(/seed\s*4242/)
    expect(run).toMatch(/4 Provider call\(s\)/)
    expect(screen.getByRole('button', { name: /Open the new Dataset Version/ })).toBeInTheDocument()
  })

  it('ends the step with the Checks panel, and Label waits for the acknowledgement', async () => {
    const user = userEvent.setup()
    const pii = {
      id: 'c1',
      kind: 'pii_found',
      severity: 'warning' as const,
      message: 'Possible PII in column email.',
      subject_type: 'dataset_version' as const,
      subject_id: 'v2',
      details: { column: 'email' },
      acknowledged: false,
      acknowledged_at: null,
      note: null,
    }
    const acked = new Set<string>()
    await runToCompletion(user, {
      'GET /jobs/j1': () => ({ id: 'j1', status: 'running', progress: { done: 1, total: 4 }, result: null }),
      'GET /checks': () => {
        const check = { ...pii, acknowledged: acked.has('c1'), acknowledged_at: '2024-01-01T00:00:00Z' }
        return { checks: [check], unacknowledged_warnings: acked.has('c1') ? 0 : 1 }
      },
      'POST /checks/c1/acknowledge': () => {
        acked.add('c1')
        return { ...pii, acknowledged: true, acknowledged_at: '2024-01-01T00:00:00Z' }
      },
    })

    const es = FakeEventSource.instances.at(-1)!
    es.emit('completed', {
      status: 'completed',
      progress: { done: 4, total: 4 },
      result: RUN_RESULT,
      error: null,
    })

    // the gate is on the version the run just produced
    expect(await screen.findByText(pii.message)).toBeInTheDocument()
    const gate = screen.getByRole('button', { name: 'Continue to Label' })
    expect(gate).toBeDisabled()

    await user.click(screen.getByRole('button', { name: 'Acknowledge' }))
    await user.click(screen.getByRole('button', { name: 'Record Acknowledgement' }))
    await waitFor(() => expect(gate).toBeEnabled())
    await user.click(gate)
    // the Label step is where the acknowledgement sends you
    await waitFor(() => expect(screen.queryByText('Ready to generate')).not.toBeInTheDocument())
    expect(screen.getByText('Jev Questions')).toBeInTheDocument()
  })

  it('says so when a run dropped rows, rather than quietly losing them', async () => {
    const user = userEvent.setup()
    await runToCompletion(user, {
      'GET /jobs/j1': () => ({ id: 'j1', status: 'running', progress: {}, result: null }),
    })
    FakeEventSource.instances.at(-1)!.emit('completed', {
      status: 'completed',
      progress: {},
      result: { ...RUN_RESULT, dropped_rows: 37 },
      error: null,
    })
    expect(await screen.findByText(/37 rows were dropped/)).toBeInTheDocument()
  })

  it('a cancelled run does not open the gate to Label', async () => {
    const user = userEvent.setup()
    await runToCompletion(user, {
      'GET /jobs/j1': () => ({ id: 'j1', status: 'running', progress: {}, result: null }),
    })
    FakeEventSource.instances.at(-1)!.emit('cancelled', {
      status: 'cancelled',
      progress: {},
      result: null,
      error: null,
    })
    await waitFor(() =>
      expect(screen.queryByRole('button', { name: 'Continue to Label' })).not.toBeInTheDocument(),
    )
  })
})

describe('Generate step — options', () => {
  it('explains each Generation Mode in plain language and defaults to hybrid', async () => {
    mockFetch(baseHandlers)
    renderStep()
    expect(await screen.findByRole('radio', { name: /hybrid/ })).toBeChecked()
    expect(screen.getByText(/Gaussian copula/)).toBeInTheDocument()
    expect(screen.getByText(/no Provider calls at all/)).toBeInTheDocument()
    expect(screen.getByText(/costs a Provider call/)).toBeInTheDocument()
  })

  it('offers a Provider override, marking the unavailable ones', async () => {
    mockFetch(baseHandlers)
    renderStep()
    const select = (await screen.findByLabelText('Provider (optional)')) as HTMLSelectElement
    expect(within(select).getByRole('option', { name: 'agent default' })).toBeInTheDocument()
    // the query resolves after the first paint, so wait for the list to land
    await waitFor(() => expect(select.options).toHaveLength(3))
    const codex = within(select).getByRole('option', { name: /codex/ }) as HTMLOptionElement
    expect(codex).toBeDisabled()
    expect(codex).toHaveTextContent('CLI not found on PATH')
  })

  it('keeps the seed and Balance Targets out of the way until asked for', async () => {
    const user = userEvent.setup()
    mockFetch(baseHandlers)
    renderStep()
    expect(await screen.findByLabelText('Seed (optional)')).toBeInTheDocument()
    expect(screen.queryByLabelText('Categorical column')).not.toBeInTheDocument()

    await user.click(await screen.findByRole('button', { name: /Add Balance Targets/ }))
    expect(screen.getByLabelText('Categorical column')).toBeInTheDocument()
  })

  it('nudges toward uploading when the project has no Dataset Versions', async () => {
    mockFetch({ ...baseHandlers, 'GET /projects/p1/dataset_versions': () => [] })
    renderStep()
    expect(
      await screen.findByText(/No Dataset Versions yet.*Upload one on the Dataset tab/s),
    ).toBeInTheDocument()
  })
})

describe('Column Spec wire format', () => {
  it('omits blanks rather than sending nulls the API would reject', () => {
    expect(
      toWireSpec({ name: 'age', type: 'number', min: null, max: null, categories: null, description: null }),
    ).toEqual({ name: 'age', type: 'number' })
    expect(
      toWireSpec({ name: 'plan', type: 'categorical', min: null, max: null, categories: ['a', 'b'], description: 'plan tier' }),
    ).toEqual({ name: 'plan', type: 'categorical', categories: ['a', 'b'], description: 'plan tier' })
  })

  it('reads a form row back into a spec, keeping blanks as null', () => {
    expect(toSpec({ name: ' age ', type: 'nonsense', min: '', max: '18' }, 0)).toEqual({
      name: 'age',
      type: 'number',
      min: null,
      max: 18,
      categories: null,
      description: null,
    })
    expect(toSpec({ name: 'plan', type: 'categorical', categories: 'a, b ,, ' }, 0).categories).toEqual(['a', 'b'])
  })
})

describe('provider options failure', () => {
  it('does not break the step when the Provider list cannot be read', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const full = String(input).replace(/^\/api/, '')
        if (full.startsWith('/providers/options')) return new Response('', { status: 500 })
        const table: Record<string, unknown> = {
          '/projects': [project],
          '/projects/p1': project,
          '/projects/p1/dataset_versions': [V1],
          '/checks': { checks: [], unacknowledged_warnings: 0 },
        }
        return json(table[full.split('?')[0]] ?? [])
      }),
    )
    renderStep()
    // the default option is still there; only the list is missing
    const select = await screen.findByLabelText('Provider (optional)')
    expect(within(select).getByRole('option', { name: 'agent default' })).toBeInTheDocument()
  })
})
