import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import App from '@/App'
import { json, mockFetch, renderWithProviders } from '@/test/render'

const PROJECTS = [
  {
    id: 'p1',
    name: 'Churn Lab',
    created_at: new Date().toISOString(),
    version_count: 3,
    latest_version_id: 'v3',
  },
]

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('App shell', () => {
  it('shows the sidebar with primary navigation', async () => {
    mockFetch({ 'GET /projects': () => PROJECTS })
    renderWithProviders(<App />)
    expect(screen.getByRole('link', { name: 'Dat Distiller' })).toBeInTheDocument()
    expect(screen.getByRole('navigation', { name: 'Primary' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Settings' })).toBeInTheDocument()
    expect(await screen.findByRole('navigation', { name: 'Projects' })).toHaveTextContent('Churn Lab')
  })

  it('lists projects on the landing page and opens one', async () => {
    const user = userEvent.setup()
    mockFetch({
      'GET /projects': () => PROJECTS,
      'GET /projects/p1': () => PROJECTS[0],
      'GET /projects/p1/dataset_versions': () => [],
    })
    renderWithProviders(<App />)
    const main = screen.getByRole('main')
    await user.click(await within(main).findByRole('link', { name: 'Churn Lab' }))
    await waitFor(() => expect(screen.getByRole('heading', { name: 'Churn Lab' })).toBeInTheDocument())
    for (const tab of ['Dataset', 'Generate', 'Label', 'Train']) {
      expect(screen.getByRole('tab', { name: tab })).toBeInTheDocument()
    }
  })

  it('creates a project through the dialog', async () => {
    const user = userEvent.setup()
    mockFetch({
      'GET /projects': () => [],
      'GET /projects/p9': () => ({ ...PROJECTS[0], id: 'p9', name: 'Fresh' }),
      'GET /projects/p9/dataset_versions': () => [],
      'POST /projects': () => ({ ...PROJECTS[0], id: 'p9', name: 'Fresh' }),
    })
    renderWithProviders(<App />)
    await user.click(await screen.findByRole('button', { name: 'New Project' }))
    await user.type(screen.getByLabelText('Name'), 'Fresh')
    await user.click(screen.getByRole('button', { name: 'Create' }))
    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Fresh' })).toBeInTheDocument(),
    )
  })

  it('surfaces a name conflict inline in the create dialog', async () => {
    const user = userEvent.setup()
    vi.stubGlobal(
      'fetch',
      vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
        const method = (init?.method ?? 'GET').toUpperCase()
        if (method === 'POST') return json({ detail: 'project name exists' }, 409)
        return json([], 200)
      }),
    )
    renderWithProviders(<App />)
    await user.click(await screen.findByRole('button', { name: 'New Project' }))
    await user.type(screen.getByLabelText('Name'), 'Dupe')
    await user.click(screen.getByRole('button', { name: 'Create' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('already exists')
  })

  it('asks for confirmation before deleting a project', async () => {
    const user = userEvent.setup()
    mockFetch({ 'GET /projects': () => PROJECTS, 'DELETE /projects/p1': () => null })
    renderWithProviders(<App />)
    const main = screen.getByRole('main')
    const card = await within(main).findByText('Churn Lab')
    const deleteButton = card
      .closest('div.group')!
      .querySelector('button')!
    await user.click(deleteButton)
    expect(await screen.findByRole('heading', { name: /Delete “Churn Lab”/ })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Delete project' }))
    await waitFor(() =>
      expect(screen.queryByRole('heading', { name: /Delete “Churn Lab”/ })).not.toBeInTheDocument(),
    )
  })

  it('says so when deleting a project fails, and keeps the dialog open', async () => {
    // `remove.mutateAsync` rejects. With nothing catching it, the dialog stayed
    // open, no error was rendered and an unhandled rejection escaped — so a
    // destructive action that failed looked exactly like a button that did
    // nothing, and since the Project is still there the answer is to try again.
    const user = userEvent.setup()
    const unhandled: unknown[] = []
    const onUnhandled = (event: PromiseRejectionEvent) => {
      unhandled.push(event.reason)
      event.preventDefault()
    }
    window.addEventListener('unhandledrejection', onUnhandled)
    try {
      mockFetch({
        'GET /projects': () => PROJECTS,
        'DELETE /projects/p1': () => json({ detail: 'database is locked' }, 500),
      })
      renderWithProviders(<App />)
      const card = await within(screen.getByRole('main')).findByText('Churn Lab')
      await user.click(card.closest('div.group')!.querySelector('button')!)
      await user.click(await screen.findByRole('button', { name: 'Delete project' }))

      const alert = await screen.findByRole('alert')
      expect(alert).toHaveTextContent('database is locked')
      // Still open, so the Project is visibly still there.
      expect(screen.getByRole('heading', { name: /Delete .Churn Lab./ })).toBeInTheDocument()
      expect(unhandled).toEqual([])
    } finally {
      window.removeEventListener('unhandledrejection', onUnhandled)
    }
  })

  it('routes /settings to the settings page', async () => {
    mockFetch({
      'GET /projects': () => [],
      'GET /settings': () => ({
        default_provider: 'openrouter',
        models: {},
        soft_limits: { upload_rows: 10000, generation_rows: 5000, labeling_calls: 200 },
        review_threshold: 0.8,
        fairness_gap_threshold: 0.1,
        keys: { openrouter: { set: true, source: 'env' }, typesafe: { set: false, source: null } },
        providers: [{ id: 'openrouter', kind: 'http', available: true }],
      }),
      'GET /health': () => ({ status: 'ok', version: '0.1.0', extras: {} }),
    })
    renderWithProviders(<App />, { route: '/settings' })
    expect(await screen.findByRole('heading', { name: 'Settings' })).toBeInTheDocument()
    expect(await screen.findByText('set (env)')).toBeInTheDocument()
    expect(screen.getByText('not set')).toBeInTheDocument()
  })

  it('shows the Training Runs list where the Train step will grow', async () => {
    mockFetch({
      'GET /projects': () => PROJECTS,
      'GET /projects/p1': () => PROJECTS[0],
      'GET /projects/p1/dataset_versions': () => [],
      'GET /train/runs': () => ({ project_id: 'p1', count: 0, runs: [] }),
    })
    renderWithProviders(<App />, { route: '/projects/p1/train' })
    expect(await screen.findByText('Training Runs')).toBeInTheDocument()
    // the card header renders while the query is still in flight
    expect(await screen.findByText(/No Training Runs yet/)).toBeInTheDocument()
  })

  it('reports a missing project with a way back', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        if (String(input).includes('/projects/')) return json({ detail: 'no such project' }, 404)
        return json([], 200)
      }),
    )
    renderWithProviders(<App />, { route: '/projects/nope/dataset' })
    expect(await screen.findByText('Project not found.')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Back to Projects' })).toBeInTheDocument()
  })
})
