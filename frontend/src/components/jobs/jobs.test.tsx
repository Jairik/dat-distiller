import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { JobButton } from '@/components/jobs/job-button'
import { JobProgress } from '@/components/jobs/job-progress'
import { RunningJobsBanner } from '@/components/jobs/running-jobs-banner'
import { json, mockFetch, renderWithProviders } from '@/test/render'
import { stubEventSource, waitForEventSource } from '@/test/sse'

/** EventSource double: captures instances so tests can push SSE frames. */

beforeEach(() => {
  stubEventSource()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

const jobDto = {
  id: 'j1',
  type: 'generate',
  project_id: 'p1',
  status: 'running',
  resumable: false,
  params: {},
  progress: { done: 2, total: 10 },
  result: null,
  error: null,
  created_at: new Date().toISOString(),
}

describe('JobProgress', () => {
  it('shows live progress from SSE and a result when done', async () => {
    mockFetch({ 'GET /jobs/j1': () => jobDto })
    renderWithProviders(
      <JobProgress
        jobId="j1"
        label="Generation"
        renderResult={(r) => <span>made {String(r.n)} rows</span>}
      />,
    )
    expect(await screen.findByText('Generation')).toBeInTheDocument()
    expect(screen.getByText('running')).toBeInTheDocument()

    const es = await waitForEventSource('j1')
    es.emit('progress', { status: 'running', progress: { done: 7, total: 10 }, result: null, error: null })
    expect(await screen.findByText('70%')).toBeInTheDocument()

    es.emit('completed', {
      status: 'completed',
      progress: { done: 10, total: 10 },
      result: { n: 42 },
      error: null,
    })
    expect(await screen.findByText(/made 42 rows/)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Cancel' })).not.toBeInTheDocument()
  })

  it('cancels a running job', async () => {
    const user = userEvent.setup()
    const { calls } = mockFetch({
      'GET /jobs/j1': () => jobDto,
      'POST /jobs/j1/cancel': () => ({ ...jobDto, status: 'cancelled' }),
    })
    renderWithProviders(<JobProgress jobId="j1" label="Run" />)
    await screen.findByText('Run')
    await user.click(screen.getByRole('button', { name: 'Cancel' }))
    await waitFor(() => expect(calls).toContainEqual(['POST', '/jobs/j1/cancel']))
  })

  it('falls back to polling when the SSE stream errors', async () => {
    mockFetch({
      'GET /jobs/j1': () => ({ ...jobDto, progress: { done: 3, total: 10 } }),
    })
    renderWithProviders(<JobProgress jobId="j1" label="Poll me" />)
    const es = await waitForEventSource('j1')
    es.onerror?.({})
    expect(await screen.findByText('30%')).toBeInTheDocument()
    expect(es.closed).toBe(true)
  })

  it('says the state is unknown when the job cannot be read, instead of waiting forever', async () => {
    // A 500 on the job detail, a pruned job row, or a server that went away
    // mid-run: all of these used to render "Waiting for …" with no error and no
    // way to tell, which told someone to leave the page and come back to a run
    // that may not exist.
    mockFetch({ 'GET /jobs/j1': () => json({ detail: 'job store is locked' }, 500) })
    renderWithProviders(<JobProgress jobId="j1" label="Generation" />)

    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent('could not be read')
    expect(alert).toHaveTextContent('unknown')
    // The server's own reason, not a bare status code.
    expect(alert).toHaveTextContent('job store is locked')
    expect(screen.queryByText(/Waiting for/)).not.toBeInTheDocument()
  })

  it('offers a retry, and recovers when the job can be read again', async () => {
    let fail = true
    mockFetch({
      'GET /jobs/j1': () =>
        fail ? json({ detail: 'job store is locked' }, 500) : { ...jobDto, progress: { done: 5, total: 10 } },
    })
    const user = userEvent.setup()
    renderWithProviders(<JobProgress jobId="j1" label="Generation" />)
    await screen.findByRole('alert')

    fail = false
    await user.click(screen.getByRole('button', { name: 'Try again' }))
    expect(await screen.findByText('50%')).toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })
})

describe('JobButton', () => {
  it('starts a job and shows the result with a run-again affordance', async () => {
    const user = userEvent.setup()
    mockFetch({
      'POST /run': () => ({ id: 'j2' }),
      'GET /jobs/j2': () => ({ ...jobDto, id: 'j2', status: 'completed', progress: { done: 5, total: 5 }, result: { ok: true } }),
    })
    renderWithProviders(
      <JobButton label="Check data" start={() => fetch('/api/run', { method: 'POST' }).then((r) => r.json())} renderResult={() => 'all good'} />,
    )
    await user.click(screen.getByRole('button', { name: 'Check data' }))
    expect(await screen.findByText(/all good/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Run again' })).toBeInTheDocument()
  })

  it('surfaces a start failure inline', async () => {
    const user = userEvent.setup()
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => json({ detail: 'no Provider configured' }, 422)),
    )
    renderWithProviders(
      <JobButton label="Go" start={() => fetch('/api/run', { method: 'POST' }).then((r) => (r.ok ? r.json() : Promise.reject(new Error('no Provider configured'))))} />,
    )
    await user.click(screen.getByRole('button', { name: 'Go' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('no Provider configured')
  })
})

describe('RunningJobsBanner', () => {
  it('lists active jobs across projects and links home', async () => {
    mockFetch({
      'GET /jobs': () => ({
        jobs: [
          { ...jobDto, id: 'jA', status: 'running', progress: { done: 4, total: 10 } },
          { ...jobDto, id: 'jB', status: 'completed' },
        ],
      }),
    })
    renderWithProviders(<RunningJobsBanner />)
    const bar = await screen.findByRole('status')
    expect(bar).toHaveTextContent('1 job running')
    expect(bar).toHaveTextContent('40%')
    expect(within(bar).getByRole('link', { name: /generate/i })).toHaveAttribute(
      'href',
      '/projects/p1/dataset',
    )
  })

  it('hides itself when nothing is active', async () => {
    mockFetch({ 'GET /jobs': () => ({ jobs: [{ ...jobDto, status: 'completed' }] }) })
    renderWithProviders(<RunningJobsBanner />)
    await waitFor(() => expect(screen.queryByRole('status')).not.toBeInTheDocument())
  })
})
