/**
 * ChecksPanel: the closing gate of every step. What matters here is that a
 * warning cannot be scrolled past — Continue stays shut until it is explicitly
 * acknowledged, and the acknowledged warning stays on screen afterwards.
 */

import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ChecksPanel, ChecksWarningBadge } from '@/components/checks/checks-panel'
import { checksKey } from '@/lib/checks'
import { json, mockFetch, renderWithProviders } from '@/test/render'

afterEach(() => {
  vi.unstubAllGlobals()
})

const info = {
  id: 'c-info',
  kind: 'row_count',
  severity: 'info' as const,
  message: 'Generated 500 rows from the mode sketch.',
  subject_type: 'project',
  subject_id: 'p1',
  details: { rows: 500, columns: 8 },
  acknowledged: false,
  acknowledged_at: null,
  note: null,
}

const warning = {
  id: 'c-warn',
  kind: 'pii_found',
  severity: 'warning' as const,
  message: 'Possible email addresses were found in 12 rows.',
  subject_type: 'project',
  subject_id: 'p1',
  details: { columns: ['email'], rows: 12 },
  acknowledged: false,
  acknowledged_at: null,
  note: null,
}

const ACKED_AT = '2024-01-01T00:00:00Z'

const body = (checks: unknown[], outstanding: number) => ({
  checks,
  unacknowledged_warnings: outstanding,
})

describe('ChecksPanel', () => {
  it('shows info and warning Checks with their severity', async () => {
    mockFetch({ 'GET /checks': () => body([info, warning], 1) })
    renderWithProviders(<ChecksPanel subjectType="project" subjectId="p1" />)

    expect(await screen.findByText(info.message)).toBeInTheDocument()
    expect(screen.getByText(warning.message)).toBeInTheDocument()
    expect(screen.getByText('pii found')).toBeInTheDocument() // kind reads as words
    expect(screen.getByText('Info')).toBeInTheDocument()
    expect(screen.getByText('Warning')).toBeInTheDocument()
  })

  it('reveals details only when the Check is expanded', async () => {
    const user = userEvent.setup()
    mockFetch({ 'GET /checks': () => body([info, warning], 1) })
    renderWithProviders(<ChecksPanel subjectType="project" subjectId="p1" />)
    await screen.findByText(info.message)

    expect(screen.queryByText(/"rows": 500/)).not.toBeInTheDocument()
    const toggle = screen.getByRole('button', { name: /Generated 500 rows/ })
    expect(toggle).toHaveAttribute('aria-expanded', 'false')
    await user.click(toggle)
    expect(await screen.findByText(/"rows": 500/)).toBeInTheDocument()
    expect(toggle).toHaveAttribute('aria-expanded', 'true')
  })

  it('never asks for an Acknowledgement on an info Check', async () => {
    mockFetch({ 'GET /checks': () => body([info], 0) })
    renderWithProviders(<ChecksPanel subjectType="project" subjectId="p1" />)
    await screen.findByText(info.message)
    expect(screen.queryByRole('button', { name: 'Acknowledge' })).not.toBeInTheDocument()
  })

  it('says so plainly when a step raised no Checks', async () => {
    mockFetch({ 'GET /checks': () => body([], 0) })
    renderWithProviders(<ChecksPanel subjectType="project" subjectId="p1" />)
    expect(await screen.findByText(/No checks/)).toBeInTheDocument()
  })

  it('acknowledges a warning with an optional note and re-reads the gate', async () => {
    const user = userEvent.setup()
    let note: string | null = null
    const { calls } = mockFetch({
      'GET /checks': () =>
        body(
          [{ ...warning, acknowledged: note !== null, acknowledged_at: ACKED_AT, note }],
          note === null ? 1 : 0,
        ),
      'POST /checks/c-warn/acknowledge': () => {
        note = 'these addresses are synthetic'
        return { ...warning, acknowledged: true, acknowledged_at: ACKED_AT, note }
      },
    })
    renderWithProviders(
      <ChecksPanel subjectType="project" subjectId="p1" continueLabel="Start Labeling" />,
    )
    await screen.findByText(warning.message)

    await user.click(screen.getByRole('button', { name: 'Acknowledge' }))
    await user.type(
      screen.getByLabelText(/Add a note about why you are continuing/),
      'these addresses are synthetic',
    )
    await user.click(screen.getByRole('button', { name: 'Record Acknowledgement' }))

    await waitFor(() =>
      expect(calls).toContainEqual(['POST', '/checks/c-warn/acknowledge']),
    )
    // the warning stays on screen, now marked and carrying its note — it
    // belongs in the Card later
    expect(await screen.findByText('Acknowledged')).toBeInTheDocument()
    expect(screen.getByText('“these addresses are synthetic”')).toBeInTheDocument()
    expect(screen.getByText(warning.message)).toBeInTheDocument()
    // acknowledged once, the control is gone for good
    expect(screen.queryByRole('button', { name: 'Acknowledge' })).not.toBeInTheDocument()
  })

  it('reports an Acknowledgement that did not stick, and stays gated', async () => {
    const user = userEvent.setup()
    vi.stubGlobal(
      'fetch',
      vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
        const method = (init?.method ?? 'GET').toUpperCase()
        if (method === 'POST') return new Response('', { status: 500 })
        return new Response(
          JSON.stringify(body([warning], 1)),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        )
      }),
    )
    renderWithProviders(
      <ChecksPanel subjectType="project" subjectId="p1" onContinue={() => {}} />,
    )
    await screen.findByText(warning.message)
    await user.click(screen.getByRole('button', { name: 'Acknowledge' }))
    await user.click(screen.getByRole('button', { name: 'Record Acknowledgement' }))

    expect(await screen.findByRole('alert')).toHaveTextContent(/did not stick/i)
    expect(screen.getByRole('button', { name: 'Continue' })).toBeDisabled()
  })
})

describe('ChecksPanel gate', () => {
  it('keeps Continue disabled until every warning is acknowledged', async () => {
    const user = userEvent.setup()
    const second = {
      ...warning,
      id: 'c-warn-2',
      kind: 'class_imbalance',
      message: 'The target class is 4% of the rows.',
    }
    // the API is the source of truth: acknowledge one, one warning is left
    const acked = new Set<string>()
    const rows = () =>
      [warning, second].map((check) => ({
        ...check,
        acknowledged: acked.has(check.id),
        acknowledged_at: ACKED_AT,
        note: null,
      }))
    const { calls } = mockFetch({
      'GET /checks': () => {
        const all = rows()
        return body(all, all.filter((row) => !row.acknowledged).length)
      },
      'POST /checks/c-warn/acknowledge': () => {
        acked.add('c-warn')
        return rows()[0]
      },
      'POST /checks/c-warn-2/acknowledge': () => {
        acked.add('c-warn-2')
        return rows()[1]
      },
    })
    const onContinue = vi.fn()
    renderWithProviders(
      <ChecksPanel subjectType="project" subjectId="p1" onContinue={onContinue} />,
    )
    expect(await screen.findByRole('status')).toHaveTextContent('2 warnings need your Acknowledgement')

    const gate = screen.getByRole('button', { name: 'Continue' })
    expect(gate).toBeDisabled()
    expect(screen.getByText(/Acknowledge the warnings above/)).toBeInTheDocument()

    // acknowledging the first still leaves the second open, so still shut
    await user.click(screen.getAllByRole('button', { name: 'Acknowledge' })[0])
    await user.click(screen.getByRole('button', { name: 'Record Acknowledgement' }))
    await waitFor(() => expect(calls).toContainEqual(['POST', '/checks/c-warn/acknowledge']))
    expect(await screen.findByRole('status')).toHaveTextContent('1 warning needs your Acknowledgement')
    expect(screen.getByRole('button', { name: 'Continue' })).toBeDisabled()

    await user.click(screen.getByRole('button', { name: 'Acknowledge' }))
    await user.click(screen.getByRole('button', { name: 'Record Acknowledgement' }))
    await waitFor(() => expect(screen.getByRole('button', { name: 'Continue' })).toBeEnabled())
    // both warnings are still on screen, marked, once the gate opens
    expect(screen.getAllByText('Acknowledged')).toHaveLength(2)
    await user.click(screen.getByRole('button', { name: 'Continue' }))
    expect(onContinue).toHaveBeenCalledTimes(1)
  })

  it('does not claim everything was read when the Checks could not be read', async () => {
    // Two contradictory sentences used to be on screen at once: the body said
    // the Checks could not be read, and the footer said "Everything here has
    // been read". Opening the gate on a failed read is deliberate — a Check
    // never blocks the work — but that is not the same as saying there was
    // nothing to read. The e2e helper waits for exactly the old string, so it
    // could pass on an errored query.
    const onContinue = vi.fn()
    mockFetch({ 'GET /checks': () => json({ detail: 'checks store is locked' }, 500) })
    renderWithProviders(
      <ChecksPanel subjectType="project" subjectId="p1" onContinue={onContinue} />,
    )
    expect(
      await screen.findByText('The Checks could not be read, so this step is open on trust.'),
    ).toBeInTheDocument()
    expect(screen.queryByText('Everything here has been read.')).not.toBeInTheDocument()
  })

  it('unlocks immediately when the only Checks are info', async () => {
    const user = userEvent.setup()
    const onContinue = vi.fn()
    mockFetch({ 'GET /checks': () => body([info], 0) })
    renderWithProviders(
      <ChecksPanel subjectType="project" subjectId="p1" onContinue={onContinue} />,
    )
    await screen.findByText(info.message)
    expect(await screen.findByText('Everything here has been read.')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Continue' }))
    expect(onContinue).toHaveBeenCalledTimes(1)
  })

  it('stays out of the way when the Checks cannot be read', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(JSON.stringify({ detail: 'boom' }), { status: 500 })),
    )
    renderWithProviders(
      <ChecksPanel subjectType="project" subjectId="p1" onContinue={() => {}} />,
    )
    expect(
      await screen.findByText(/nothing to acknowledge/i),
    ).toBeInTheDocument()
    // a Check never blocks the work itself, so the gate is not held hostage
    expect(await screen.findByRole('button', { name: 'Continue' })).toBeEnabled()
  })

  it('renders no footer gate when the caller has no Continue of its own', async () => {
    mockFetch({ 'GET /checks': () => body([warning], 1) })
    renderWithProviders(<ChecksPanel subjectType="project" subjectId="p1" />)
    await screen.findByText(warning.message)
    expect(screen.queryByRole('button', { name: 'Continue' })).not.toBeInTheDocument()
  })

  it('asks for no subject and skips the request when the subject is not known yet', () => {
    const { calls } = mockFetch({ 'GET /checks': () => body([], 0) })
    renderWithProviders(<ChecksPanel subjectType="dataset_version" subjectId={undefined} />)
    expect(calls).toHaveLength(0)
  })
})

describe('ChecksWarningBadge', () => {
  it('counts the open warnings of its subject', async () => {
    mockFetch({ 'GET /checks': () => body([info, warning], 1) })
    renderWithProviders(<ChecksWarningBadge subjectType="project" subjectId="p1" />)
    expect(await screen.findByText(/1 warning to acknowledge/)).toBeInTheDocument()
  })

  it('is quiet at zero, and stays quiet while the count is still unknown', async () => {
    mockFetch({ 'GET /checks': () => body([info], 0) })
    const { container } = renderWithProviders(
      <ChecksWarningBadge subjectType="project" subjectId="p1" />,
    )
    // a header must not flash a badge in before the count has landed
    expect(container).toHaveTextContent('')
    await waitFor(() => expect(container).toHaveTextContent(''))
    expect(screen.queryByText(/to acknowledge/)).not.toBeInTheDocument()
  })
})

describe('checksKey', () => {
  it('scopes the cache per subject so two steps never share a gate', () => {
    expect(checksKey('project', 'p1')).toEqual(['checks', 'project', 'p1'])
    expect(checksKey('training_run', 'r1')).not.toEqual(checksKey('project', 'r1'))
  })
})
