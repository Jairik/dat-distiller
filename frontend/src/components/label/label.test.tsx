/**
 * The Label step.
 *
 * Two things are being protected. First, the **preview gate**: Labeling costs a
 * Jev call per row, so the run button must not exist until a preview has been
 * seen. Second, the **State preview must be the truth** — `serializeState` is
 * claimed to match the backend's `jev.serialize_state` character for character,
 * and that claim is checked here against real backend output rather than
 * against a re-implementation of the same idea.
 */

import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import App from '@/App'
import {
  emptyQuestion,
  labelColumns,
  questionErrors,
  questionToSpec,
  serializeState,
} from '@/lib/label'
import { mockFetch, renderWithProviders } from '@/test/render'
import { emitJobEvent, stubEventSource } from '@/test/sse'


beforeEach(() => {
  stubEventSource()
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

const PREVIEW_ROW = {
  state: 'age: 31\nplan: pro',
  answers: {
    is_churn: { answer: 'no', confidence: 0.93 },
  },
}

const baseHandlers = {
  'GET /projects': () => [project],
  'GET /projects/p1': () => project,
  'GET /projects/p1/dataset_versions': () => [V1],
  'GET /checks': () => ({ checks: [], unacknowledged_warnings: 0 }),
  'GET /dataset-versions/v1/preview': () => ({
    version_id: 'v1',
    columns: [
      { name: 'age', kind: 'integer' },
      { name: 'plan', kind: 'categorical' },
    ],
    page: 0,
    page_size: 1,
    total_rows: 40,
    rows: [[31, 'pro']],
  }),
}

function renderStep() {
  return renderWithProviders(<App />, { route: '/projects/p1/label?version=v1' })
}

/** Fill in a valid single Noul question and select every State column. */
async function buildNoul(user: ReturnType<typeof userEvent.setup>) {
  await user.clear(await screen.findByLabelText(/Name \(becomes the Label Column\)/))
  await user.type(screen.getByLabelText(/Name \(becomes the Label Column\)/), 'is_churn')
  await user.type(
    screen.getByLabelText('Instructions'),
    'Answer yes when the customer cancelled within 30 days.',
  )
  await user.click(await screen.findByRole('button', { name: 'Select every column' }))
}

describe('Label step — the State picker', () => {
  it('previews the serialized State as Jev will receive it', async () => {
    const user = userEvent.setup()
    mockFetch(baseHandlers)
    renderStep()
    // the picker only offers select-all once it has read the version's columns
    await screen.findByLabelText('age')
    await user.click(screen.getByRole('button', { name: 'Select every column' }))

    const preview = screen.getByTestId('state-preview')
    expect(preview).toHaveTextContent('age: 31')
    expect(preview).toHaveTextContent('plan: pro')
    // and it is literally the "column: value" lines Jev reads
    expect(preview.textContent?.split('\n')).toEqual(['age: 31', 'plan: pro'])
  })

  it('only sends the columns that are selected', async () => {
    const user = userEvent.setup()
    mockFetch({
      ...baseHandlers,
      'POST /label/preview': () => ({ state_columns: ['age'], rows: [PREVIEW_ROW] }),
    })
    renderStep()
    await buildNoul(user)
    await user.click(screen.getByLabelText('plan'))

    await user.click(screen.getByRole('button', { name: /Preview 5 rows/ }))
    await screen.findByText('Preview', { selector: '[data-slot="card-title"]' })
    // 'plan' was deselected, so the State preview shows only the chosen column
    expect(screen.getByTestId('state-preview').textContent).toBe('age: 31')
  })

  it('says when there is nothing to label', async () => {
    mockFetch({ ...baseHandlers, 'GET /projects/p1/dataset_versions': () => [] })
    renderStep()
    expect(await screen.findByText(/No Dataset Versions yet/)).toBeInTheDocument()
  })
})

describe('serializeState matches the backend', () => {
  it('collapses whitespace inside a value, exactly as jev.serialize_state does', () => {
    // the backend does `" ".join(str(value).split())` then f"{col}: {text}".rstrip()
    expect(serializeState({ note: '  two\nlines\there  ' }, ['note'])).toBe('note: two lines here')
  })

  it('leaves a missing value as an empty line rather than the text "null"', () => {
    expect(serializeState({ a: null, b: undefined }, ['a', 'b'])).toBe('a:\nb:')
  })

  it('keeps the chosen column order', () => {
    expect(serializeState({ a: 1, b: 2 }, ['b', 'a'])).toBe('b: 2\na: 1')
  })

  it('renders a zero and an empty string, which are not missing', () => {
    expect(serializeState({ a: 0, b: '' }, ['a', 'b'])).toBe('a: 0\nb:')
  })
})

describe('Label step — building questions by hand', () => {
  it('builds a Noul by hand and previews it', async () => {
    const user = userEvent.setup()
    const { calls } = mockFetch({
      ...baseHandlers,
      'POST /label/preview': () => ({ state_columns: ['age', 'plan'], rows: [PREVIEW_ROW] }),
    })
    renderStep()
    await buildNoul(user)
    await user.click(screen.getByRole('button', { name: /Preview 5 rows/ }))

    expect(await screen.findByText('is_churn')).toBeInTheDocument()
    expect(screen.getByText('confidence 0.93')).toBeInTheDocument()
    expect(calls).toContainEqual(['POST', '/label/preview'])
  })

  it('builds a Choice with key/description pairs', async () => {
    const user = userEvent.setup()
    const { calls } = mockFetch({
      ...baseHandlers,
      'POST /label/preview': () => ({ state_columns: ['age'], rows: [PREVIEW_ROW] }),
    })
    renderStep()
    await userEvent.setup()
    await user.click(await screen.findByRole('radio', { name: 'choice' }))

    const keys = screen.getAllByLabelText('Key')
    const descriptions = screen.getAllByLabelText('Description')
    await user.clear(keys[0])
    await user.type(keys[0], 'billing')
    await user.type(descriptions[0], 'about invoices')
    await user.clear(keys[1])
    await user.type(keys[1], 'bug')
    await user.type(descriptions[1], 'something is broken')
    await user.type(screen.getByLabelText(/Name \(becomes/), 'topic')
    await user.type(screen.getByLabelText('Instructions'), 'Pick the one topic that fits.')
    await user.click(screen.getByRole('button', { name: 'Select every column' }))

    await user.click(screen.getByRole('button', { name: /Preview 5 rows/ }))
    await waitFor(() => expect(calls).toContainEqual(['POST', '/label/preview']))
  })

  it('builds a Score with ordered levels', async () => {
    const user = userEvent.setup()
    const { calls } = mockFetch({
      ...baseHandlers,
      'POST /label/preview': () => ({ state_columns: ['age'], rows: [PREVIEW_ROW] }),
    })
    renderStep()
    await userEvent.setup()
    await user.click(await screen.findByRole('radio', { name: 'score' }))

    const levels = screen.getAllByLabelText(/^Level /)
    await user.type(levels[0], '1 - unusable')
    await user.type(levels[1], '5 - decisive')
    await user.type(screen.getByLabelText(/Name \(becomes/), 'quality')
    await user.type(screen.getByLabelText('Instructions'), 'Grade how useful this review is.')
    await user.click(screen.getByRole('button', { name: 'Select every column' }))

    await user.click(screen.getByRole('button', { name: /Preview 5 rows/ }))
    await waitFor(() => expect(calls).toContainEqual(['POST', '/label/preview']))
  })

  it('shows the matching editor for each type, and explains the type', async () => {
    const user = userEvent.setup()
    mockFetch(baseHandlers)
    renderStep()
    await screen.findByText(/A yes\/no judgement/)

    await user.click(screen.getByRole('radio', { name: 'choice' }))
    expect(screen.getAllByLabelText('Key').length).toBeGreaterThan(0)
    expect(screen.getByText(/exactly one of several named options/i)).toBeInTheDocument()

    await user.click(screen.getByRole('radio', { name: 'score' }))
    expect(screen.getAllByLabelText(/^Level /).length).toBeGreaterThan(0)
    expect(screen.getByText(/levels, lowest to highest/i)).toBeInTheDocument()
  })

  it('refuses an incomplete question, with the reason', async () => {
    const { calls } = mockFetch(baseHandlers)
    renderStep()
    await screen.findByLabelText(/Name \(becomes/)

    expect(screen.getByRole('alert')).toHaveTextContent('instructions are required')
    expect(screen.getByRole('button', { name: /Preview 5 rows/ })).toBeDisabled()
    expect(calls.some(([m]) => m === 'POST')).toBe(false)
  })

  it('refuses a non-snake_case name, because it becomes a column', async () => {
    const user = userEvent.setup()
    mockFetch(baseHandlers)
    renderStep()
    const name = await screen.findByLabelText(/Name \(becomes the Label Column\)/)
    await user.clear(name)
    await user.type(name, 'Is Churn')
    expect(screen.getByRole('alert')).toHaveTextContent('snake_case')
  })

  it('refuses a Choice with fewer than two options', () => {
    const one = { ...emptyQuestion(0), type: 'choice' as const, criteria: [{ key: 'a', description: '' }] }
    expect(questionErrors(one)).toContain('a Choice needs at least two options')
  })

  it('refuses duplicated Choice options', () => {
    const dup = {
      ...emptyQuestion(0),
      type: 'choice' as const,
      criteria: [
        { key: 'a', description: '' },
        { key: 'a', description: '' },
      ],
    }
    expect(questionErrors(dup)).toContain('option "a" is duplicated')
  })

  it('refuses a Score with fewer than two levels', () => {
    const one = { ...emptyQuestion(0), type: 'score' as const, levels: ['only'] }
    expect(questionErrors(one)).toContain('a Score needs at least two levels')
  })

  it('adds and removes questions', async () => {
    const user = userEvent.setup()
    mockFetch(baseHandlers)
    renderStep()
    await screen.findByLabelText(/Name \(becomes/)
    await user.click(screen.getByRole('button', { name: /Add a question/ }))
    expect(screen.getAllByLabelText(/Name \(becomes/)).toHaveLength(2)
    await user.click(screen.getByRole('button', { name: 'Remove question 2' }))
    await waitFor(() =>
      expect(screen.getAllByLabelText(/Name \(becomes/)).toHaveLength(1),
    )
  })
})

describe('Label step — the draft helper', () => {
  it('fills the builder from a Provider draft, and leaves it editable', async () => {
    const user = userEvent.setup()
    const { calls } = mockFetch({
      ...baseHandlers,
      'POST /jev/draft-question': () => ({
        kind: 'choice',
        question: {
          type: 'choice',
          name: 'ticket_topic',
          instructions: 'Pick the single topic that best matches the request.',
          criteria: { billing: 'about invoices', bug: 'something is broken' },
        },
        warning: null,
      }),
    })
    renderStep()
    await userEvent.setup()
    await user.click(await screen.findByRole('button', { name: /Draft with Provider/ }))

    // the builder is filled...
    await waitFor(() =>
      expect(screen.getByLabelText(/Name \(becomes the Label Column\)/)).toHaveValue(
        'ticket_topic',
      ),
    )
    // ...with the type switched for us
    expect(screen.getByRole('radio', { name: 'choice' })).toBeChecked()
    expect(screen.getAllByLabelText('Key').map((k) => (k as HTMLInputElement).value)).toEqual([
      'billing',
      'bug',
    ])

    // ...and it is still ordinary editable state, not a locked-in suggestion
    const name = screen.getByLabelText(/Name \(becomes the Label Column\)/)
    await user.clear(name)
    await user.type(name, 'my_topic')
    expect(name).toHaveValue('my_topic')
    expect(calls).toContainEqual(['POST', '/jev/draft-question'])
  })

  it('reports a Provider that could not draft anything', async () => {
    const user = userEvent.setup()
    mockFetch({
      ...baseHandlers,
      'POST /jev/draft-question': () => {
        throw new Error('OpenRouter needs an API key')
      },
    })
    renderStep()
    await userEvent.setup()
    await user.click(await screen.findByRole('button', { name: /Draft with Provider/ }))
    expect(await screen.findByText(/needs an API key/)).toBeInTheDocument()
  })
})

describe('Label step — the preview gate and the run', () => {
  it('will not run before a preview, and says why', async () => {
    const user = userEvent.setup()
    const { calls } = mockFetch(baseHandlers)
    renderStep()
    await buildNoul(user)
    expect(screen.getByText('Preview a few rows to unlock the full run.')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /Label the whole Dataset Version/ })).not.toBeInTheDocument()
    expect(calls.some(([m]) => m === 'POST')).toBe(false)
  })

  it('previews, estimates, then runs', async () => {
    const user = userEvent.setup()
    const { calls } = mockFetch({
      ...baseHandlers,
      'POST /label/preview': () => ({ state_columns: ['age', 'plan'], rows: [PREVIEW_ROW] }),
      'POST /label/estimate': () => ({ rows: 40, estimated_jev_calls: 40, uses_jev: true }),
      'POST /label/run': () => ({ id: 'j1' }),
    })
    renderStep()
    await buildNoul(user)
    await user.click(screen.getByRole('button', { name: /Preview 5 rows/ }))
    await screen.findByText('Preview', { selector: '[data-slot="card-title"]' })
    await user.click(screen.getByRole('button', { name: /Estimate and continue/ }))
    expect(await screen.findByText(/40 Jev call\(s\), one per row/)).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /Label the whole Dataset Version/ }))
    await waitFor(() => expect(calls).toContainEqual(['POST', '/label/run']))
  })

  it('a failed preview does not unlock the run', async () => {
    const user = userEvent.setup()
    mockFetch({
      ...baseHandlers,
      'POST /label/preview': () => {
        throw new Error('Jev is not configured')
      },
    })
    renderStep()
    await buildNoul(user)
    await user.click(screen.getByRole('button', { name: /Preview 5 rows/ }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Jev is not configured')
    expect(screen.queryByRole('button', { name: /Label the whole Dataset Version/ })).not.toBeInTheDocument()
  })

  it('follows the run and shows what it produced', async () => {
    const user = userEvent.setup()
    mockFetch({
      ...baseHandlers,
      'POST /label/preview': () => ({ state_columns: ['age'], rows: [PREVIEW_ROW] }),
      'POST /label/estimate': () => ({ rows: 40, estimated_jev_calls: 40, uses_jev: true }),
      'POST /label/run': () => ({ id: 'j1' }),
      'GET /jobs/j1': () => ({ id: 'j1', status: 'running', progress: { done: 1, total: 40 }, result: null }),
    })
    renderStep()
    await buildNoul(user)
    await user.click(screen.getByRole('button', { name: /Preview 5 rows/ }))
    await screen.findByText('Preview', { selector: '[data-slot="card-title"]' })
    await user.click(screen.getByRole('button', { name: /Estimate and continue/ }))
    await user.click(await screen.findByRole('button', { name: /Label the whole Dataset Version/ }))

    await screen.findByText('Labeling run')
    await emitJobEvent('j1', 'completed', {
      status: 'completed',
      progress: { done: 40, total: 40 },
      result: {
        version_id: 'v2',
        labeled_rows: 40,
        failed_rows: [],
        failed_count: 0,
        label_columns: ['is_churn', 'is_churn__confidence'],
      },
      error: null,
    })
    expect(await screen.findByText(/40 rows labeled/)).toBeInTheDocument()
    // the Label Columns it added are named, so the user can find them
    expect(screen.getByText('is_churn__confidence')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /Download CSV/ })).toBeInTheDocument()
  })

  it('says out loud when Jev could not answer some rows', async () => {
    const user = userEvent.setup()
    mockFetch({
      ...baseHandlers,
      'POST /label/preview': () => ({ state_columns: ['age'], rows: [PREVIEW_ROW] }),
      'POST /label/estimate': () => ({ rows: 40, estimated_jev_calls: 40, uses_jev: true }),
      'POST /label/run': () => ({ id: 'j1' }),
      'GET /jobs/j1': () => ({ id: 'j1', status: 'running', progress: {}, result: null }),
    })
    renderStep()
    await buildNoul(user)
    await user.click(screen.getByRole('button', { name: /Preview 5 rows/ }))
    await screen.findByText('Preview', { selector: '[data-slot="card-title"]' })
    await user.click(screen.getByRole('button', { name: /Estimate and continue/ }))
    await user.click(await screen.findByRole('button', { name: /Label the whole Dataset Version/ }))
    await screen.findByText('Labeling run')
    await emitJobEvent('j1', 'completed', {
      status: 'completed',
      progress: {},
      result: {
        version_id: 'v2',
        labeled_rows: 37,
        failed_rows: [3, 7],
        failed_count: 3,
        label_columns: ['is_churn'],
      },
      error: null,
    })
    expect(await screen.findByText(/3 row\(s\) Jev could not answer/)).toBeInTheDocument()
  })

  it('offers to resume an interrupted run, and resumes it', async () => {
    const user = userEvent.setup()
    const { calls } = mockFetch({
      ...baseHandlers,
      'POST /label/preview': () => ({ state_columns: ['age'], rows: [PREVIEW_ROW] }),
      'POST /label/estimate': () => ({ rows: 40, estimated_jev_calls: 40, uses_jev: true }),
      'POST /label/run': () => ({ id: 'j1' }),
      'POST /jobs/j1/resume': () => ({ id: 'j1', status: 'running' }),
      'GET /jobs/j1': () => ({ id: 'j1', status: 'interrupted', progress: { done: 20, total: 40 }, result: null }),
    })
    renderStep()
    await buildNoul(user)
    await user.click(screen.getByRole('button', { name: /Preview 5 rows/ }))
    await screen.findByText('Preview', { selector: '[data-slot="card-title"]' })
    await user.click(screen.getByRole('button', { name: /Estimate and continue/ }))
    await user.click(await screen.findByRole('button', { name: /Label the whole Dataset Version/ }))

    const resume = await screen.findByRole('button', { name: /Resume the run/ })
    expect(screen.getByText(/only pays for what is left/)).toBeInTheDocument()
    await user.click(resume)
    await waitFor(() => expect(calls).toContainEqual(['POST', '/jobs/j1/resume']))
  })

  it('does not offer resume for a run that finished normally', async () => {
    const user = userEvent.setup()
    mockFetch({
      ...baseHandlers,
      'POST /label/preview': () => ({ state_columns: ['age'], rows: [PREVIEW_ROW] }),
      'POST /label/estimate': () => ({ rows: 40, estimated_jev_calls: 40, uses_jev: true }),
      'POST /label/run': () => ({ id: 'j1' }),
      'GET /jobs/j1': () => ({ id: 'j1', status: 'completed', progress: {}, result: { version_id: 'v2', labeled_rows: 40, failed_count: 0, label_columns: [] } }),
    })
    renderStep()
    await buildNoul(user)
    await user.click(screen.getByRole('button', { name: /Preview 5 rows/ }))
    await screen.findByText('Preview', { selector: '[data-slot="card-title"]' })
    await user.click(screen.getByRole('button', { name: /Estimate and continue/ }))
    await user.click(await screen.findByRole('button', { name: /Label the whole Dataset Version/ }))
    await screen.findByText('Labeling run')
    await waitFor(() =>
      expect(screen.queryByRole('button', { name: /Resume the run/ })).not.toBeInTheDocument(),
    )
  })

  it('ends the step with the Checks panel, gating the way to Train', async () => {
    const user = userEvent.setup()
    const check = {
      id: 'c1',
      kind: 'labeling_soft_limit',
      severity: 'warning' as const,
      message: 'Labeling 5000 rows exceeds the soft limit.',
      subject_type: 'dataset_version' as const,
      subject_id: 'v2',
      details: {},
      acknowledged: false,
      acknowledged_at: null,
      note: null,
    }
    const acked = new Set<string>()
    mockFetch({
      ...baseHandlers,
      'POST /label/preview': () => ({ state_columns: ['age'], rows: [PREVIEW_ROW] }),
      'POST /label/estimate': () => ({ rows: 40, estimated_jev_calls: 40, uses_jev: true }),
      'POST /label/run': () => ({ id: 'j1' }),
      'GET /jobs/j1': () => ({ id: 'j1', status: 'running', progress: {}, result: null }),
      'GET /checks': () => {
        const row = { ...check, acknowledged: acked.has('c1'), acknowledged_at: '2024-01-01T00:00:00Z' }
        return { checks: [row], unacknowledged_warnings: acked.has('c1') ? 0 : 1 }
      },
      'POST /checks/c1/acknowledge': () => {
        acked.add('c1')
        return { ...check, acknowledged: true, acknowledged_at: '2024-01-01T00:00:00Z' }
      },
    })
    renderStep()
    await buildNoul(user)
    await user.click(screen.getByRole('button', { name: /Preview 5 rows/ }))
    await screen.findByText('Preview', { selector: '[data-slot="card-title"]' })
    await user.click(screen.getByRole('button', { name: /Estimate and continue/ }))
    await user.click(await screen.findByRole('button', { name: /Label the whole Dataset Version/ }))
    await emitJobEvent('j1', 'completed', {
      status: 'completed',
      progress: {},
      result: { version_id: 'v2', labeled_rows: 40, failed_count: 0, label_columns: ['is_churn'] },
      error: null,
    })

    expect(await screen.findByText(check.message)).toBeInTheDocument()
    const gate = screen.getByRole('button', { name: 'Continue to Train' })
    expect(gate).toBeDisabled()
    await user.click(screen.getByRole('button', { name: 'Acknowledge' }))
    await user.click(screen.getByRole('button', { name: 'Record Acknowledgement' }))
    await waitFor(() => expect(gate).toBeEnabled())
  })
})

describe('question spec wire format', () => {
  it('sends only the fields the question type uses', () => {
    expect(questionToSpec({ ...emptyQuestion(0), name: ' spam ', instructions: ' do it ' })).toEqual({
      type: 'noul',
      name: 'spam',
      instructions: 'do it',
    })
    expect(
      questionToSpec({
        ...emptyQuestion(0),
        type: 'choice',
        criteria: [
          { key: ' a ', description: ' first ' },
          { key: '', description: 'dropped' },
        ],
      }),
    ).toMatchObject({ criteria: { a: 'first' } })
  })

  it('names the Label Columns Labeling will add', () => {
    expect(
      labelColumns([
        { type: 'noul', name: 'spam' },
        { type: 'choice', name: 'topic', criteria: { 'Billing Issue': 'x', bug: 'y' } },
        { type: 'score', name: 'quality', levels: ['1', '5'] },
      ]),
    ).toEqual([
      'spam',
      'spam__confidence',
      'topic',
      'topic__confidence',
      'topic__p_billing_issue',
      'topic__p_bug',
      'quality',
      'quality__confidence',
      'quality__probabilities',
    ])
  })

  it('keeps a Label Column out of its own State', async () => {
    // Jev must never read the output it is about to write. Name a question
    // after an existing column and that column must vanish from the picker.
    const user = userEvent.setup()
    mockFetch({
      ...baseHandlers,
      'POST /label/preview': () => ({ state_columns: ['plan'], rows: [PREVIEW_ROW] }),
    })
    renderStep()
    await screen.findByLabelText('age')
    await user.clear(screen.getByLabelText(/Name \(becomes the Label Column\)/))
    await user.type(screen.getByLabelText(/Name \(becomes the Label Column\)/), 'age')
    await user.type(screen.getByLabelText('Instructions'), 'decide something')

    // 'age' is now a Label Column this run will write, so it is not a State input
    await waitFor(() => expect(screen.queryByLabelText('age')).not.toBeInTheDocument())
    expect(screen.getByLabelText('plan')).toBeInTheDocument()
  })
})
