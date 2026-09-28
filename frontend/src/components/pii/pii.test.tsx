/**
 * The PII findings panel.
 *
 * The acceptance criterion this file is really about is the first one: **example
 * matches must never display raw PII.** The examples arrive already masked from
 * the backend, and this suite checks the rendered DOM for the raw values as well
 * as for the masked ones — asserting only that the mask is present would pass
 * just as happily if the raw value sat right next to it.
 */

import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import App from '@/App'
import { examplesOf, piiDetailsOf, type PiiFinding } from '@/lib/pii'
import { mockFetch, renderWithProviders } from '@/test/render'

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
  name: 'PII Lab',
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
    { name: 'email', kind: 'categorical' },
    { name: 'note', kind: 'text' },
  ],
  provenance_summary: { uploaded: 40 },
  seed: null,
  meta: {},
  created_at: '2024-01-01T00:00:00Z',
}

const SCAN = {
  version_id: 'v1',
  findings: [
    {
      column: 'email',
      detector: 'email',
      count: 12,
      // masked by the backend — this is what must reach the screen
      example_cells: ['j***@example.com', 'b***.com'],
      row_examples: [0, 3],
    },
    {
      column: 'note',
      detector: 'phone',
      count: 2,
      example_cells: ['call 5***'],
      row_examples: [7],
    },
  ] as PiiFinding[],
  summary: { columns: ['email', 'note'], detectors: { email: 12, phone: 2 }, total_findings: 14 },
  actions: ['mask', 'drop'],
}

const PII_CHECK = {
  id: 'c1',
  kind: 'pii_found',
  severity: 'warning' as const,
  message: "Possible PII in column 'email' (email hit(s) of email, 1 detector(s))",
  subject_type: 'dataset_version' as const,
  subject_id: 'v1',
  details: { column: 'email', detector: 'email', detectors: { email: 12 }, count: 12, examples: ['j***@example.com'] },
  acknowledged: false,
  acknowledged_at: null,
  note: null,
}

const baseHandlers = {
  'GET /projects': () => [project],
  'GET /projects/p1': () => project,
  'GET /projects/p1/dataset_versions': () => [V1],
  'GET /dataset-versions/v1/preview': () => ({
    version_id: 'v1',
    columns: [
      { name: 'email', kind: 'categorical' },
      { name: 'note', kind: 'text' },
    ],
    page: 0,
    page_size: 25,
    total_rows: 40,
    rows: [['jane@example.com', 'call 555-123-4567']],
  }),
  'GET /dataset-versions/v1/pii': () => SCAN,
  'GET /checks': () => ({ checks: [PII_CHECK], unacknowledged_warnings: 1 }),
}

function renderPanel() {
  return renderWithProviders(<App />, { route: '/projects/p1/dataset?v=v1' })
}

describe('PiiPanel — findings', () => {
  it('lists the flagged columns with their detectors and match counts', async () => {
    mockFetch(baseHandlers)
    renderPanel()
    await screen.findByText('PII findings')

    const email = await screen.findByTestId('pii-email')
    expect(within(email).getByText('12 match(es)')).toBeInTheDocument()
    // the column is named, and the detector that fired is labelled
    const names = within(email).getAllByText('email')
    expect(names).toHaveLength(2)  // the column name and the detector badge
    expect(email.textContent).toContain('email')

    const note = screen.getByTestId('pii-note')
    expect(within(note).getByText('phone')).toBeInTheDocument()
    expect(within(note).getByText('2 match(es)')).toBeInTheDocument()
  })

  it('shows masked examples and never the raw values', async () => {
    mockFetch(baseHandlers)
    renderPanel()
    const email = await screen.findByTestId('pii-email')

    expect(within(email).getByText('j***@example.com')).toBeInTheDocument()
    expect(within(email).getByText('b***.com')).toBeInTheDocument()
    // the raw values must not appear anywhere on the page — the preview table
    // legitimately shows them, so this is scoped to the findings panel
    expect(email.textContent).not.toContain('jane@example.com')
  })

  it('says so when a finding has no examples, rather than fetching the real ones', async () => {
    const { calls } = mockFetch({
      ...baseHandlers,
      'GET /dataset-versions/v1/pii': () => ({
        ...SCAN,
        findings: [{ ...SCAN.findings[0], example_cells: [] }],
      }),
    })
    renderPanel()
    const email = await screen.findByTestId('pii-email')
    expect(within(email).getByText(/No examples were kept/)).toBeInTheDocument()
    // crucially, no extra request went out to re-read the column
    expect(
      calls.filter(([, p]) => p.includes('preview') || p.includes('download')).length,
    ).toBeLessThan(3)
  })

  it('renders nothing at all when the version is clean', async () => {
    mockFetch({
      ...baseHandlers,
      'GET /dataset-versions/v1/pii': () => ({
        ...SCAN,
        findings: [],
        summary: { columns: [], detectors: {}, total_findings: 0 },
      }),
    })
    renderPanel()
    await screen.findByText('Version 1')
    expect(screen.queryByText('PII findings')).not.toBeInTheDocument()
  })

  it('stays quiet rather than shouting when the scan cannot be read', async () => {
    mockFetch({
      ...baseHandlers,
      'GET /dataset-versions/v1/pii': () => {
        throw new Error('boom')
      },
    })
    renderPanel()
    expect(await screen.findByText(/The scan could not be read/)).toBeInTheDocument()
  })
})

describe('PiiPanel — mask and drop', () => {
  it('applies a chosen action and reports what it did', async () => {
    const user = userEvent.setup()
    const { calls } = mockFetch({
      ...baseHandlers,
      'POST /dataset-versions/v1/pii/actions': () => ({
        version: { id: 'v2', parent_id: 'v1', number: 2, origin: 'redacted' },
        summary: {
          columns: { email: { action: 'mask', replacements: 12 } },
          masked: ['email'],
          dropped: [],
          noops: [],
        },
        remaining_findings: { columns: [], detectors: {}, total_findings: 0 },
      }),
    })
    renderPanel()
    const email = await screen.findByTestId('pii-email')

    await user.click(within(email).getByRole('button', { name: 'Mask' }))
    expect(within(email).getByText(/rewrites the detected spans as \[REDACTED\]/)).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /Apply to 1 column/ }))
    await waitFor(() => expect(calls).toContainEqual(['POST', '/dataset-versions/v1/pii/actions']))
    const done = await screen.findByText(/A new Dataset Version was created/)
    expect(done).toHaveTextContent('in the history')
    expect(done).toHaveTextContent('"masked":["email"]')
  })

  it('explains what dropping will do before it does it', async () => {
    const user = userEvent.setup()
    mockFetch(baseHandlers)
    renderPanel()
    const note = await screen.findByTestId('pii-note')

    await user.click(within(note).getByRole('button', { name: 'Drop' }))
    expect(within(note).getByText(/removes this whole column/)).toBeInTheDocument()
  })

  it('can act on more than one column at once', async () => {
    const user = userEvent.setup()
    const { calls } = mockFetch({
      ...baseHandlers,
      'POST /dataset-versions/v1/pii/actions': () => ({
        version: { id: 'v2', parent_id: 'v1', number: 2, origin: 'redacted' },
        summary: { columns: {}, masked: ['email'], dropped: ['note'], noops: [] },
        remaining_findings: { columns: [], detectors: {}, total_findings: 0 },
      }),
    })
    renderPanel()
    await screen.findByTestId('pii-email')
    await user.click(within(screen.getByTestId('pii-email')).getByRole('button', { name: 'Mask' }))
    await user.click(within(screen.getByTestId('pii-note')).getByRole('button', { name: 'Drop' }))
    await user.click(screen.getByRole('button', { name: /Apply to 2 columns/ }))
    await waitFor(() => expect(calls).toContainEqual(['POST', '/dataset-versions/v1/pii/actions']))
  })

  it('cannot apply until a column has been chosen', async () => {
    mockFetch(baseHandlers)
    renderPanel()
    await screen.findByTestId('pii-email')
    expect(screen.getByRole('button', { name: /^Apply to/ })).toBeDisabled()
    expect(screen.getByText(/Pick Mask or Drop on a column/)).toBeInTheDocument()
  })

  it('reports a refused action, and keeps the choices', async () => {
    const user = userEvent.setup()
    mockFetch({
      ...baseHandlers,
      'POST /dataset-versions/v1/pii/actions': () => {
        throw new Error('pii action must be one of')
      },
    })
    renderPanel()
    await screen.findByTestId('pii-email')
    await user.click(within(screen.getByTestId('pii-email')).getByRole('button', { name: 'Mask' }))
    await user.click(screen.getByRole('button', { name: /Apply to 1 column/ }))

    expect(await screen.findByRole('alert')).toHaveTextContent('pii action must be one of')
    expect(screen.getByRole('button', { name: /Apply to 1 column/ })).toBeEnabled()
  })

  it('makes the chosen action the pressed one, so it cannot be misread', async () => {
    const user = userEvent.setup()
    mockFetch(baseHandlers)
    renderPanel()
    const email = await screen.findByTestId('pii-email')
    await user.click(within(email).getByRole('button', { name: 'Drop' }))
    // clicking Mask then Drop settles on Drop
    await user.click(within(email).getByRole('button', { name: 'Mask' }))
    expect(within(email).getByText(/rewrites the detected spans/)).toBeInTheDocument()
  })
})

describe('PiiPanel — continue (the warn path)', () => {
  it('acknowledges the Check rather than changing any data', async () => {
    const user = userEvent.setup()
    let acked = false
    const { calls } = mockFetch({
      ...baseHandlers,
      'GET /checks': () => ({
        checks: [{ ...PII_CHECK, acknowledged: acked, acknowledged_at: acked ? '2024-01-01T00:00:00Z' : null }],
        unacknowledged_warnings: acked ? 0 : 1,
      }),
      'POST /checks/c1/acknowledge': () => {
        acked = true
        return { ...PII_CHECK, acknowledged: true, acknowledged_at: '2024-01-01T00:00:00Z' }
      },
    })
    renderPanel()
    await screen.findByText('PII findings')

    await user.click(screen.getByRole('button', { name: /Continue as they are/ }))
    await waitFor(() => expect(calls).toContainEqual(['POST', '/checks/c1/acknowledge']))
    // the warn path creates no Dataset Version at all
    expect(calls.some(([, p]) => p.includes('pii/actions'))).toBe(false)
    expect(await screen.findByRole('button', { name: 'Acknowledged' })).toBeInTheDocument()
  })

  it('cannot continue when there is nothing outstanding to acknowledge', async () => {
    mockFetch({
      ...baseHandlers,
      'GET /checks': () => ({ checks: [{ ...PII_CHECK, acknowledged: true }], unacknowledged_warnings: 0 }),
    })
    renderPanel()
    await screen.findByText('PII findings')
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /Continue as they are/ })).toBeDisabled(),
    )
    expect(screen.getByText(/already acknowledged/)).toBeInTheDocument()
  })

  it('reports an acknowledgement that did not stick', async () => {
    const user = userEvent.setup()
    mockFetch({
      ...baseHandlers,
      'POST /checks/c1/acknowledge': () => {
        throw new Error('not found')
      },
    })
    renderPanel()
    await screen.findByText('PII findings')
    await user.click(screen.getByRole('button', { name: /Continue as they are/ }))
    expect(await screen.findByRole('alert')).toHaveTextContent('not found')
  })
})

describe('pii helpers', () => {
  it('reads Check details defensively', () => {
    expect(piiDetailsOf({ column: 'email', count: 3 })).toMatchObject({ column: 'email', count: 3 })
    expect(piiDetailsOf({})).toBeNull()
    expect(piiDetailsOf(null)).toBeNull()
    expect(piiDetailsOf('nope')).toBeNull()
    expect(piiDetailsOf({ column: 'email' })?.examples).toEqual([])
  })

  it('never invents an example', () => {
    expect(examplesOf({ ...SCAN.findings[0], example_cells: [] })).toEqual([])
    expect(
      examplesOf({ ...SCAN.findings[0], example_cells: ['', 'ok', 'fine'] as string[] }),
    ).toEqual(['ok', 'fine'])
  })
})

describe('PiiPanel — the staged PII Actions belong to one Dataset Version', () => {
  it('does not carry a staged PII Action over to another version', async () => {
    // Selecting a version only changes `?v=`, so nothing unmounts. Without a
    // key the panel keeps `choice`, and the Mask staged against v1 is then
    // applied to v2 — a silent rewrite of a Dataset Version the user never
    // looked at.
    const user = userEvent.setup()
    const V2 = { ...V1, id: 'v2', number: 2, origin: 'masked', parent_id: 'v1' }
    mockFetch({
      ...baseHandlers,
      'GET /projects/p1': () => ({ ...project, version_count: 2, latest_version_id: 'v2' }),
      'GET /projects/p1/dataset_versions': () => [V1, V2],
      'GET /dataset-versions/v2/pii': () => ({
        ...SCAN,
        version_id: 'v2',
        findings: [{ ...SCAN.findings[0] }],
        summary: { columns: ['email'], detectors: { email: 12 }, total_findings: 12 },
      }),
      'GET /dataset-versions/v2/preview': () => ({
        ...baseHandlers['GET /dataset-versions/v1/preview'](),
        version_id: 'v2',
      }),
    })
    renderWithProviders(<App />, { route: '/projects/p1/dataset?v=v1' })

    await user.click(within(await screen.findByTestId('pii-email')).getByRole('button', { name: 'Mask' }))
    expect(await screen.findByText('email: mask')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Apply to 1 column/ })).toBeEnabled()

    // Move to the other version in the tree.
    await user.click(screen.getByRole('button', { name: /^v2/ }))

    // The staged action belonged to v1 and must not be offered for v2.
    await waitFor(() =>
      expect(screen.getByText('Pick Mask or Drop on a column to act on it.')).toBeInTheDocument(),
    )
    expect(screen.queryByText('email: mask')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Apply to 0 columns/ })).toBeDisabled()
  })
})
