/**
 * Cards: viewed from the history panel and from the Training Runs list, and
 * downloadable.
 *
 * The test that matters is not "a drawer opened" — it is that the Card is
 * rendered as *what it says* rather than as a wall of raw Markdown. A Card's
 * whole job is to be read, and a reader who gets `| # Dataset Card — v3 |`
 * instead of a heading has been handed a file, not a record.
 */

import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import App from '@/App'
import { Markdownish, cardUrl, inline } from '@/components/cards/card-viewer'
import { formatPrimary } from '@/lib/runs'
import { mockFetch, renderWithProviders, text } from '@/test/render'

beforeEach(() => {
  vi.stubGlobal('EventSource', class {
    listeners = new Map<string, unknown>()
    addEventListener() {}
    close() {}
    onerror: unknown = null
  } as never)
})

afterEach(() => {
  vi.unstubAllGlobals()
})

const project = {
  id: 'p1',
  name: 'Card Lab',
  created_at: '2024-01-01T00:00:00Z',
  version_count: 2,
  latest_version_id: 'v2',
}

const V1 = {
  id: 'v1',
  project_id: 'p1',
  parent_id: null,
  number: 1,
  origin: 'uploaded',
  row_count: 40,
  columns: [{ name: 'age', kind: 'integer' }],
  provenance_summary: { uploaded: 40 },
  seed: 7,
  meta: {},
  created_at: '2024-01-01T00:00:00Z',
}

const V2 = { ...V1, id: 'v2', number: 2, provenance_summary: { synthetic: 40 } }

const DATASET_CARD = `# Dataset Card — v2

- **Dataset Version**: \`v2\`
- **Rows**: 40
- **Seed**: 7

## Provenance

| Where the rows came from | Rows |
|---|---|
| synthetic | 40 |

## Jev Questions

| Name | Type | Instructions |
|---|---|---|
| \`is_churn\` | noul | Answer yes when the customer left. |

## Checks and Acknowledgements

| Check | Severity | Message | Acknowledged |
|---|---|---|---|
| \`pii_found\` | warning | Possible PII in column 'note' | **no** |

## Known limitations

- _Rows written by a Provider cannot be reproduced exactly._
`

const MODEL_CARD = `# Model Card — Training Run \`run1\`

- **Target**: \`is_churn\`
- **Task Type**: classification

## Leaderboard

| Rank | Model | Primary |
|---|---|---|
| 1 | Logistic Regression | 0.9120 |

## Known limitations

- _The leaderboard differences sit within the noise of that split size._
`

const RUN = {
  training_run_id: 'run1',
  job_id: 'run1',
  project_id: 'p1',
  status: 'completed',
  progress: {},
  created_at: '2024-01-02T00:00:00Z',
  finished_at: '2024-01-02T00:01:00Z',
  version_id: 'v2',
  target: 'is_churn',
  task_type: 'classification',
  seed: 99,
  primary_metric: 'f1_macro',
  n_models: 7,
  best_model: 'logistic_regression',
  best_primary_value: 0.912,
  warnings: [],
  error: null,
}

const baseHandlers = {
  'GET /projects': () => [project],
  'GET /projects/p1': () => project,
  'GET /projects/p1/dataset_versions': () => [V1, V2],
  'GET /dataset-versions/v2/preview': () => ({
    version_id: 'v2',
    columns: [{ name: 'age', kind: 'integer' }],
    page: 0,
    page_size: 25,
    total_rows: 40,
    rows: [[31]],
  }),
  'GET /dataset-versions/v2/pii': () => ({
    version_id: 'v2',
    findings: [],
    summary: { columns: [], detectors: {}, total_findings: 0 },
    actions: ['mask', 'drop'],
  }),
  'GET /checks': () => ({ checks: [], unacknowledged_warnings: 0 }),
}

describe('Dataset Cards — from the history panel', () => {
  it('opens from the selected version and renders it as a document', async () => {
    const user = userEvent.setup()
    mockFetch({
      ...baseHandlers,
      'GET /dataset-versions/v2/card': () => text(DATASET_CARD, 200, 'text/markdown'),
    })
    renderWithProviders(<App />, { route: '/projects/p1/dataset?v=v2' })

    await user.click(await screen.findByRole('button', { name: 'Dataset Card' }))
    const body = await screen.findByTestId('card-body')

    // a heading, not a pipe character
    expect(
      within(body).getByRole('heading', { name: 'Dataset Card — v2' }),
    ).toBeInTheDocument()
    expect(within(body).getByRole('heading', { name: 'Provenance' })).toBeInTheDocument()
    expect(body.textContent).not.toContain('|---')
  })

  it('renders tables as tables, with the header row', async () => {
    const user = userEvent.setup()
    mockFetch({ ...baseHandlers, 'GET /dataset-versions/v2/card': () => text(DATASET_CARD, 200, 'text/markdown') })
    renderWithProviders(<App />, { route: '/projects/p1/dataset?v=v2' })
    await user.click(await screen.findByRole('button', { name: 'Dataset Card' }))

    const body = await screen.findByTestId('card-body')
    // the Card has several tables; the first is the Provenance one
    const tables = await within(body).findAllByRole('table')
    const provenance = tables[0]
    const headers = within(provenance).getAllByRole('columnheader').map((h) => h.textContent)
    expect(headers).toEqual(['Where the rows came from', 'Rows'])
    // the |---|---| separator is not a data row
    expect(within(provenance).queryByText('---')).not.toBeInTheDocument()
    expect(within(provenance).getByText('synthetic')).toBeInTheDocument()
    // one body row, not two: the separator must not have become data
    expect(within(provenance).getAllByRole('row')).toHaveLength(2)
  })

  it('shows the raw Markdown on request, and the exact text is downloadable', async () => {
    const user = userEvent.setup()
    mockFetch({ ...baseHandlers, 'GET /dataset-versions/v2/card': () => text(DATASET_CARD, 200, 'text/markdown') })
    renderWithProviders(<App />, { route: '/projects/p1/dataset?v=v2' })
    await user.click(await screen.findByRole('button', { name: 'Dataset Card' }))

    await user.click(await screen.findByRole('tab', { name: 'Markdown' }))
    const raw = await screen.findByTestId('card-raw')
    expect(raw.textContent).toBe(DATASET_CARD)
    expect(raw.textContent).toContain('|---|---|')

    // and the download is the same document
    const link = screen.getByRole('link', { name: /Download \.md/ })
    expect(link).toHaveAttribute('href', '/api/dataset-versions/v2/card?format=markdown')
    expect(link).toHaveAttribute('download', 'dataset-card.md')
  })

  it('does not read a Card until it is asked for', async () => {
    const { calls } = mockFetch({ ...baseHandlers, 'GET /dataset-versions/v2/card': () => text(DATASET_CARD, 200, 'text/markdown') })
    renderWithProviders(<App />, { route: '/projects/p1/dataset?v=v2' })
    await screen.findByText('Version 2')
    await new Promise((r) => setTimeout(r, 60))
    // the history panel can hold many versions; firing one request per row to
    // build buttons nobody clicked would be rude
    expect(calls.some(([, p]) => p.includes('/card'))).toBe(false)
  })

  it('reports a Card it could not read, and does not show an empty document', async () => {
    const user = userEvent.setup()
    mockFetch({
      ...baseHandlers,
      'GET /dataset-versions/v2/card': () => {
        throw new Error('not found')
      },
    })
    renderWithProviders(<App />, { route: '/projects/p1/dataset?v=v2' })
    await user.click(await screen.findByRole('button', { name: 'Dataset Card' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Could not read this Card')
    expect(screen.queryByRole('table')).not.toBeInTheDocument()
  })

  it('renders the reproducibility caveat rather than swallowing it', async () => {
    const user = userEvent.setup()
    mockFetch({ ...baseHandlers, 'GET /dataset-versions/v2/card': () => text(DATASET_CARD, 200, 'text/markdown') })
    renderWithProviders(<App />, { route: '/projects/p1/dataset?v=v2' })
    await user.click(await screen.findByRole('button', { name: 'Dataset Card' }))
    const body = await screen.findByTestId('card-body')
    expect(body.textContent).toContain('cannot be reproduced exactly')
    // an acknowledged Check is still shown, and visibly unacknowledged
    expect(body.textContent).toContain('pii_found')
  })
})

describe('Model Cards — from the Training Runs list', () => {
  it('opens a Model Card for a finished run', async () => {
    const user = userEvent.setup()
    mockFetch({
      ...baseHandlers,
      'GET /train/runs': () => ({ project_id: 'p1', count: 1, runs: [RUN] }),
      'GET /train/runs/run1/card': () => text(MODEL_CARD, 200, 'text/markdown'),
    })
    renderWithProviders(<App />, { route: '/projects/p1/train' })

    await user.click(await screen.findByRole('button', { name: 'Model Card' }))
    const body = await screen.findByTestId('card-body')
    expect(
      within(body).getByRole('heading', { name: 'Model Card — Training Run run1' }),
    ).toBeInTheDocument()
    expect(within(body).getByRole('heading', { name: 'Leaderboard' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /Download \.md/ })).toHaveAttribute(
      'href',
      '/api/train/runs/run1/card?format=markdown',
    )
  })

  it('summarises the run so you know which Card you are opening', async () => {
    mockFetch({ ...baseHandlers, 'GET /train/runs': () => ({ project_id: 'p1', count: 1, runs: [RUN] }) })
    renderWithProviders(<App />, { route: '/projects/p1/train' })
    const row = await screen.findByTestId('training-run-run1')
    expect(within(row).getByText('is_churn')).toBeInTheDocument()
    expect(within(row).getByText('classification')).toBeInTheDocument()
    expect(row.textContent).toContain('logistic_regression')
    expect(row.textContent).toContain('f1_macro 0.9120')
    expect(row.textContent).toContain('seed 99')
  })

  it('arrows the primary metric in the direction that metric actually goes', async () => {
    // The list always printed an up arrow, because `formatPrimary` defaults to
    // higher-is-better and the summary never said otherwise. For a run whose
    // primary metric is rmse or mae, scrolling the history told the reader that
    // the bigger number was the better one.
    mockFetch({
      ...baseHandlers,
      'GET /train/runs': () => ({
        project_id: 'p1',
        count: 1,
        runs: [
          {
            ...RUN,
            task_type: 'regression',
            primary_metric: 'rmse',
            primary_metric_higher_is_better: false,
            best_primary_value: 1.0,
          },
        ],
      }),
    })
    renderWithProviders(<App />, { route: '/projects/p1/train' })
    const row = await screen.findByTestId('training-run-run1')
    expect(row.textContent).toContain('rmse 1.0000 ↓')
    expect(row.textContent).not.toContain('rmse 1.0000 ↑')
  })

  it('lists a failed run with its reason rather than hiding it', async () => {
    mockFetch({
      ...baseHandlers,
      'GET /train/runs': () => ({
        project_id: 'p1',
        count: 1,
        runs: [{ ...RUN, status: 'failed', error: 'the Target had one class' }],
      }),
    })
    renderWithProviders(<App />, { route: '/projects/p1/train' })
    const row = await screen.findByTestId('training-run-run1')
    expect(within(row).getByText('the Target had one class')).toBeInTheDocument()
    // and its Card is still reachable — a failure is part of the record
    expect(within(row).getByRole('button', { name: 'Model Card' })).toBeInTheDocument()
  })

  it('says plainly when nothing has been trained yet', async () => {
    mockFetch({ ...baseHandlers, 'GET /train/runs': () => ({ project_id: 'p1', count: 0, runs: [] }) })
    renderWithProviders(<App />, { route: '/projects/p1/train' })
    expect(await screen.findByText(/No Training Runs yet/)).toBeInTheDocument()
  })

  it('surfaces the run warnings', async () => {
    mockFetch({
      ...baseHandlers,
      'GET /train/runs': () => ({
        project_id: 'p1',
        count: 1,
        runs: [{ ...RUN, warnings: ['the Target is 4% of the rows'] }],
      }),
    })
    renderWithProviders(<App />, { route: '/projects/p1/train' })
    const row = await screen.findByTestId('training-run-run1')
    expect(within(row).getByText('the Target is 4% of the rows')).toBeInTheDocument()
  })
})

describe('the Markdown renderer', () => {
  it('drops the table separator instead of showing it as a row', () => {
    const { container } = render(<Markdownish text={'| A | B |\n|---|---|\n| 1 | 2 |'} />)
    expect(container.querySelectorAll('tbody tr')).toHaveLength(1)
    expect(container.textContent).not.toContain('---')
  })

  it('handles a table with no separator line', () => {
    const { container } = render(<Markdownish text={'| A | B |\n| 1 | 2 |'} />)
    expect(container.querySelectorAll('tbody tr')).toHaveLength(1)
  })

  it('groups a run of bullet lines into one list', () => {
    const { container } = render(<Markdownish text={'- one\n- two\n- three'} />)
    expect(container.querySelectorAll('ul')).toHaveLength(1)
    expect(container.querySelectorAll('li')).toHaveLength(3)
  })

  it('renders inline code, bold and links', () => {
    const { container } = render(<Markdownish text={'a `code` and **bold** and [x](y)'} />)
    expect(container.querySelector('code')?.textContent).toBe('code')
    expect(container.querySelector('strong')?.textContent).toBe('bold')
    expect(container.querySelector('a')?.getAttribute('href')).toBe('y')
  })

  it('never throws on a stray pipe or unclosed marker', () => {
    expect(() => render(<Markdownish text={'| a | b\n**unclosed `code'} />)).not.toThrow()
  })

  it('builds the card URL from the kind', () => {
    expect(cardUrl({ kind: 'dataset', id: 'v1', title: 'x' })).toBe(
      '/api/dataset-versions/v1/card?format=markdown',
    )
    expect(cardUrl({ kind: 'model', id: 'r1', title: 'x' }, 'json')).toBe(
      '/api/train/runs/r1/card?format=json',
    )
  })
})

describe('inline()', () => {
  it('leaves plain text alone', () => {
    render(<>{inline('just words')}</>)
    expect(screen.getByText('just words')).toBeInTheDocument()
  })
})

describe('formatPrimary', () => {
  it('shows an arrow in the direction that is better', () => {
    expect(formatPrimary(0.912, true)).toBe('0.9120 ↑')
    expect(formatPrimary(1.5, false)).toBe('1.5000 ↓')
  })

  it('shows an em dash, not a zero, when no Model could be ranked', () => {
    expect(formatPrimary(null)).toBe('—')
  })
})
