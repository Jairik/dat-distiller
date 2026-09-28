/**
 * The Review Queue.
 *
 * Reviewing is the tedium in this app, so the thing being protected here is
 * speed without accidental damage: the keyboard must be able to drive a whole
 * queue, and it must go quiet the moment the reviewer starts typing a value —
 * otherwise `a` in an override field would accept the row instead of typing.
 */

import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import App from '@/App'
import { overrideProblem, type QueueItem } from '@/lib/review'
import { mockFetch, renderWithProviders } from '@/test/render'
import { stubEventSource } from '@/test/sse'


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
  version_count: 2,
  latest_version_id: 'v2',
}

const V1: Record<string, unknown> = {
  id: 'v1',
  project_id: 'p1',
  parent_id: null,
  number: 1,
  origin: 'uploaded',
  row_count: 40,
  columns: [{ name: 'age', kind: 'integer' }],
  provenance_summary: { uploaded: 40 },
  seed: null,
  meta: {},
  created_at: '2024-01-01T00:00:00Z',
}

const V2: typeof V1 = { ...V1, id: 'v2', number: 2, origin: 'labeled', provenance_summary: { jev: 40 } }

function item(index: number, overrides: Partial<QueueItem> = {}): QueueItem {
  return {
    row_index: index,
    family: 'tone',
    question_type: 'choice',
    answer: 'pos',
    confidence: 0.4,
    decision: null,
    override: null,
    note: null,
    ...overrides,
  }
}

const QUEUE = {
  version_id: 'v2',
  threshold: 0.8,
  families: { tone: 'choice' },
  items: [item(1), item(3, { confidence: 0.6 })],
  unlabeled: [],
  queued_count: 2,
  returned_count: 2,
  unreviewed_count: 2,
  decisions: ['accept', 'override', 'exclude'],
}

const STATUS = {
  version_id: 'v2',
  threshold: 0.8,
  queued_count: 2,
  unlabeled_count: 0,
  unreviewed_count: 2,
}

const baseHandlers = {
  'GET /projects': () => [project],
  'GET /projects/p1': () => project,
  'GET /projects/p1/dataset_versions': () => [V1, V2],
  'GET /dataset-versions/v2/preview': () => ({
    version_id: 'v2',
    columns: [{ name: 'age', kind: 'integer' }],
    page: 0,
    page_size: 1,
    total_rows: 40,
    rows: [[31]],
  }),
  'GET /dataset-versions/v2/review-queue': () => QUEUE,
  'GET /dataset-versions/v2/review-status': () => STATUS,
  'GET /checks': () => ({ checks: [], unacknowledged_warnings: 0 }),
}

function renderQueue() {
  return renderWithProviders(<App />, { route: '/projects/p1/dataset?v=v2' })
}

describe('ReviewQueuePanel', () => {
  it('lists the queued labels with the State, the answer and the confidence', async () => {
    mockFetch(baseHandlers)
    renderQueue()
    await screen.findByText('Review Queue')

    expect(await screen.findByTestId('queue-item-1-tone')).toBeInTheDocument()
    const card = screen.getByTestId('queue-item-1-tone')
    expect(within(card).getByText('tone')).toBeInTheDocument()
    expect(within(card).getByText('choice')).toBeInTheDocument()
    expect(within(card).getByText('row 1')).toBeInTheDocument()
    expect(within(card).getByText('Jev said')).toBeInTheDocument()
    expect(within(card).getByText('confidence 0.40')).toBeInTheDocument()
  })

  it('shows progress through the queue', async () => {
    mockFetch(baseHandlers)
    renderQueue()
    await screen.findByTestId('queue-item-1-tone')
    expect(screen.getByText('Reviewing 1 of 2')).toBeInTheDocument()
  })

  it('only appears on a labeled version', async () => {
    mockFetch({ ...baseHandlers, 'GET /projects/p1/dataset_versions': () => [V1] })
    renderWithProviders(<App />, { route: '/projects/p1/dataset?v=v1' })
    await screen.findByText('Version 1')
    expect(screen.queryByText('Review Queue')).not.toBeInTheDocument()
  })

  it('says plainly when the queue is empty', async () => {
    mockFetch({
      ...baseHandlers,
      'GET /dataset-versions/v2/review-queue': () => ({ ...QUEUE, items: [], queued_count: 0 }),
      'GET /dataset-versions/v2/review-status': () => ({ ...STATUS, unreviewed_count: 0, queued_count: 0 }),
    })
    renderQueue()
    expect(await screen.findByText(/Nothing to review at this threshold/)).toBeInTheDocument()
  })

  it('warns about labels Jev never answered, which cannot be reviewed', async () => {
    mockFetch({
      ...baseHandlers,
      'GET /dataset-versions/v2/review-queue': () => ({
        ...QUEUE,
        unlabeled: [{ row_index: 5, family: 'tone', question_type: 'choice' }],
      }),
      'GET /dataset-versions/v2/review-status': () => ({ ...STATUS, unlabeled_count: 1, unreviewed_count: 3 }),
    })
    renderQueue()
    expect(await screen.findByText(/1 label\(s\) were never answered/)).toBeInTheDocument()
    expect(screen.getByText(/cannot be/)).toHaveTextContent(/nothing was ever proposed/)
  })
})

describe('Review Queue — the threshold slider', () => {
  it('re-reads the queue at the new threshold and shows the live count', async () => {
    const user = userEvent.setup()
    const { calls } = mockFetch({
      ...baseHandlers,
      'GET /dataset-versions/v2/review-queue': () => QUEUE,
      'GET /dataset-versions/v2/review-status': () => ({ ...STATUS, unreviewed_count: 5 }),
    })
    renderQueue()
    await screen.findByTestId('queue-item-1-tone')
    // the count comes from its own (cheap) status endpoint, so it lands separately
    const count = await screen.findByText('Still outstanding')
    await waitFor(() => expect(count.parentElement).toHaveTextContent('5'))

    await user.click(screen.getByRole('slider'))
    await user.keyboard('{ArrowLeft}{ArrowLeft}')
    await waitFor(() =>
      expect(
        calls.some(([, p]) => p.includes('review-queue?threshold=') && !p.includes('threshold=0.8')),
      ).toBe(true),
    )
  })

  it('is explicit that moving the slider does not change what is trained on', async () => {
    mockFetch(baseHandlers)
    renderQueue()
    expect(await screen.findByText(/it does not change what is trained on/)).toBeInTheDocument()
  })
})

describe('Review Queue — keyboard review', () => {
  it('accepts, excludes and moves without touching the mouse', async () => {
    const user = userEvent.setup()
    const { calls } = mockFetch({
      ...baseHandlers,
      'POST /dataset-versions/v2/review': () => ({
        version: { ...V2, id: 'v3', number: 3, origin: 'reviewed', parent_id: 'v2' },
        outcome: { accepted: 1, overridden: 0, excluded_rows: [], excluded_count: 0 },
        unreviewed_count: 1,
        checks_raised: [],
      }),
    })
    renderQueue()
    await screen.findByTestId('queue-item-1-tone')

    await user.keyboard('a') // accept the first
    await waitFor(() =>
      expect(within(screen.getByTestId('queue-item-1-tone')).getByText('accept')).toBeInTheDocument(),
    )
    await user.keyboard('x') // exclude the second
    await waitFor(() =>
      expect(within(screen.getByTestId('queue-item-3-tone')).getByText('exclude')).toBeInTheDocument(),
    )
    await user.click(screen.getByRole('button', { name: /Apply 2 decisions/ }))
    await waitFor(() => expect(calls).toContainEqual(['POST', '/dataset-versions/v2/review']))
  })

  it('moves with j and k and the arrow keys', async () => {
    const user = userEvent.setup()
    mockFetch(baseHandlers)
    renderQueue()
    await screen.findByTestId('queue-item-1-tone')

    await user.keyboard('j')
    expect(await screen.findByText('Reviewing 2 of 2')).toBeInTheDocument()
    await user.keyboard('k')
    expect(await screen.findByText('Reviewing 1 of 2')).toBeInTheDocument()
    await user.keyboard('{ArrowDown}')
    expect(await screen.findByText('Reviewing 2 of 2')).toBeInTheDocument()
    // and it will not run off the end of the queue
    await user.keyboard('{ArrowDown}')
    expect(await screen.findByText('Reviewing 2 of 2')).toBeInTheDocument()
  })

  it('focuses the override field on "o", so a reviewer can type straight away', async () => {
    const user = userEvent.setup()
    mockFetch(baseHandlers)
    renderQueue()
    await screen.findByTestId('queue-item-1-tone')

    await user.keyboard('o')
    await waitFor(() =>
      expect(document.getElementById('override-1-tone')).toHaveFocus(),
    )
  })

  it('goes quiet while the reviewer is typing, so "a" types an "a"', async () => {
    const user = userEvent.setup()
    mockFetch(baseHandlers)
    renderQueue()
    await screen.findByTestId('queue-item-1-tone')

    const field = document.getElementById('override-1-tone') as HTMLInputElement
    field.focus()
    await user.keyboard('another option')
    expect(field).toHaveValue('another option')
    // typing "a" in the field must not have accepted the row
    expect(
      within(screen.getByTestId('queue-item-1-tone')).queryByText('accept'),
    ).not.toBeInTheDocument()
  })

  it('applies an override typed in the field, with Enter', async () => {
    const user = userEvent.setup()
    const { calls } = mockFetch({
      ...baseHandlers,
      'POST /dataset-versions/v2/review': () => ({
        version: { ...V2, id: 'v3', number: 3, origin: 'reviewed', parent_id: 'v2' },
        outcome: { accepted: 0, overridden: 1, excluded_rows: [], excluded_count: 0 },
        unreviewed_count: 1,
        checks_raised: [],
      }),
    })
    renderQueue()
    await screen.findByTestId('queue-item-1-tone')

    await user.type(document.getElementById('override-1-tone') as HTMLInputElement, 'neg{Enter}')
    await waitFor(() =>
      expect(within(screen.getByTestId('queue-item-1-tone')).getByText(/override/)).toBeInTheDocument(),
    )
    await user.click(screen.getByRole('button', { name: /Apply 1 decision/ }))
    await waitFor(() => expect(calls).toContainEqual(['POST', '/dataset-versions/v2/review']))
  })

  it('will not override with nothing', async () => {
    const user = userEvent.setup()
    mockFetch(baseHandlers)
    renderQueue()
    await screen.findByTestId('queue-item-1-tone')
    const overrideButton = screen.getAllByRole('button', { name: 'Override' })[0]
    expect(overrideButton).toBeDisabled()
    expect(user).toBeTruthy()
  })
})

describe('Review Queue — bulk actions and applying', () => {
  it('accepts everything shown in one click', async () => {
    const user = userEvent.setup()
    const { calls } = mockFetch({
      ...baseHandlers,
      'POST /dataset-versions/v2/review': () => ({
        version: { ...V2, id: 'v3', number: 3, origin: 'reviewed', parent_id: 'v2' },
        outcome: { accepted: 2, overridden: 0, excluded_rows: [], excluded_count: 0 },
        unreviewed_count: 0,
        checks_raised: [],
      }),
    })
    renderQueue()
    await screen.findByTestId('queue-item-1-tone')

    await user.click(screen.getByRole('button', { name: /Accept all 2 shown/ }))
    await user.click(screen.getByRole('button', { name: /Apply 2 decisions/ }))
    await waitFor(() => expect(calls).toContainEqual(['POST', '/dataset-versions/v2/review']))
  })

  it('clears staged decisions without applying', async () => {
    const user = userEvent.setup()
    const { calls } = mockFetch(baseHandlers)
    renderQueue()
    await screen.findByTestId('queue-item-1-tone')

    await user.click(screen.getAllByRole('button', { name: 'Accept' })[0])
    expect(await screen.findByText('1 staged')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /Clear staged decisions/ }))
    expect(await screen.findByText('0 staged')).toBeInTheDocument()
    expect(calls.some(([m]) => m === 'POST')).toBe(false)
  })

  it('cannot apply when nothing is staged', () => {
    mockFetch(baseHandlers)
    renderQueue()
    return screen.findByTestId('queue-item-1-tone').then(() => {
      expect(screen.getByRole('button', { name: /^Apply/ })).toBeDisabled()
    })
  })

  it('says what the apply did, and offers the new Dataset Version', async () => {
    const user = userEvent.setup()
    mockFetch({
      ...baseHandlers,
      'POST /dataset-versions/v2/review': () => ({
        version: { ...V2, id: 'v3', number: 3, origin: 'reviewed', parent_id: 'v2' },
        outcome: { accepted: 1, overridden: 0, excluded_rows: [], excluded_count: 0 },
        unreviewed_count: 1,
        checks_raised: [],
      }),
    })
    renderQueue()
    await screen.findByTestId('queue-item-1-tone')
    await user.click(screen.getAllByRole('button', { name: 'Accept' })[0])
    await user.click(screen.getByRole('button', { name: /Apply 1 decision/ }))

    expect(await screen.findByRole('status')).toHaveTextContent(/The new Dataset Version is in the history/)
    expect(screen.getByRole('button', { name: /Open the reviewed Dataset Version/ })).toBeInTheDocument()
  })

  it('reports an apply that failed, and keeps the staged decisions', async () => {
    const user = userEvent.setup()
    mockFetch({
      ...baseHandlers,
      'POST /dataset-versions/v2/review': () => {
        throw new Error("'nope' is not a Label Column on this version")
      },
    })
    renderQueue()
    await screen.findByTestId('queue-item-1-tone')
    await user.click(screen.getAllByRole('button', { name: 'Accept' })[0])
    await user.click(screen.getByRole('button', { name: /Apply 1 decision/ }))

    expect(await screen.findByRole('alert')).toHaveTextContent('not a Label Column')
    // the reviewer's work is not thrown away on a failure
    expect(screen.getByText('1 staged')).toBeInTheDocument()
  })

  it('ends with the Checks panel on the reviewed version', async () => {
    const user = userEvent.setup()
    mockFetch({
      ...baseHandlers,
      'POST /dataset-versions/v2/review': () => ({
        version: { ...V2, id: 'v3', number: 3, origin: 'reviewed', parent_id: 'v2' },
        outcome: { accepted: 1, overridden: 0, excluded_rows: [], excluded_count: 0 },
        unreviewed_count: 1,
        checks_raised: [],
      }),
      'GET /checks': () => ({ checks: [], unacknowledged_warnings: 0 }),
    })
    renderQueue()
    await screen.findByTestId('queue-item-1-tone')
    await user.click(screen.getAllByRole('button', { name: 'Accept' })[0])
    await user.click(screen.getByRole('button', { name: /Apply 1 decision/ }))
    expect(await screen.findByRole('button', { name: 'Continue to Train' })).toBeInTheDocument()
  })
})

describe('overrideProblem', () => {
  it('catches values the backend would reject', () => {
    expect(overrideProblem('choice', '')).toMatch(/needs a value/)
    expect(overrideProblem('noul', 'maybe')).toMatch(/yes or no/)
    expect(overrideProblem('score', 'excellent')).toMatch(/number on the scale/)
    expect(overrideProblem('choice', 'neg')).toBeNull()
    expect(overrideProblem('noul', 'yes')).toBeNull()
    expect(overrideProblem('score', 5)).toBeNull()
  })
})

describe('ReviewQueuePanel — staged decisions belong to one Dataset Version', () => {
  it('does not carry a staged decision over to another version', async () => {
    // Selecting a version only changes `?v=`, so nothing unmounts and the panel
    // kept `staged`. The decision then went to the *new* version's endpoint with
    // the old row index — a label nobody reviewed, written into a Dataset
    // Version the reviewer was no longer looking at, branching a child from the
    // wrong parent.
    const user = userEvent.setup()
    const V3 = { ...V2, id: 'v3', number: 3, parent_id: 'v2' }
    const v3Queue = {
      ...QUEUE,
      version_id: 'v3',
      items: [item(11), item(13, { confidence: 0.6 })],
      queued_count: 2,
    }
    mockFetch({
      ...baseHandlers,
      'GET /projects/p1': () => ({ ...project, version_count: 3, latest_version_id: 'v3' }),
      'GET /projects/p1/dataset_versions': () => [V1, V2, V3],
      'GET /dataset-versions/v3/preview': () => ({
        version_id: 'v3',
        columns: [{ name: 'age', kind: 'integer' }],
        page: 0,
        page_size: 1,
        total_rows: 40,
        rows: [[44]],
      }),
      'GET /dataset-versions/v3/review-queue': () => v3Queue,
      'GET /dataset-versions/v3/review-status': () => ({ ...STATUS, version_id: 'v3' }),
    })
    renderQueue()
    await screen.findByText('Review Queue')

    // Stage an accept against v2's first row.
    await user.click(screen.getAllByRole('button', { name: 'Accept' })[0])
    expect(await screen.findByText('1 staged')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Apply 1 decision' })).toBeEnabled()

    // Move to v3 in the version tree.
    await user.click(screen.getByRole('button', { name: /^v3/ }))

    // v3's queue is showing, and the decision staged for v2 is not offered.
    await waitFor(() => expect(screen.getByTestId('queue-item-11-tone')).toBeInTheDocument())
    expect(screen.queryByText('1 staged')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Apply 0 decisions' })).toBeDisabled()
  })
})
