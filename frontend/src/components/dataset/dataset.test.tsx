import { fireEvent, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import App from '@/App'
import { mockFetch, renderWithProviders } from '@/test/render'

const project = {
  id: 'p1',
  name: 'Tree Lab',
  created_at: new Date().toISOString(),
  version_count: 4,
  latest_version_id: 'v4',
}

function version(id: string, number: number, origin: string, parent_id: string | null, summary: Record<string, number>) {
  return {
    id,
    project_id: 'p1',
    parent_id,
    number,
    origin,
    row_count: 60,
    columns: [
      { name: 'age', kind: 'integer' },
      { name: 'plan', kind: 'categorical' },
    ],
    provenance_summary: summary,
    seed: null,
    meta: {},
    created_at: new Date().toISOString(),
  }
}

const V1 = version('v1', 1, 'uploaded', null, { uploaded: 60 })
const V2 = version('v2', 2, 'generated', 'v1', { uploaded: 40, synthetic: 20 })
const V3 = version('v3', 3, 'labeled', 'v2', { uploaded: 30, synthetic: 20, jev: 10 })
const V4 = version('v4', 4, 'generated', 'v1', { uploaded: 50, synthetic: 10 })

function previewBody(page: number) {
  return {
    version_id: 'v3',
    columns: [
      { name: 'age', kind: 'integer' },
      { name: 'plan', kind: 'categorical' },
    ],
    page,
    page_size: 25,
    total_rows: 60,
    rows: Array.from({ length: 25 }, (_, i) => [20 + i, i % 2 ? 'pro' : 'basic']),
  }
}

const baseHandlers = {
  'GET /projects': () => [project],
  'GET /projects/p1': () => project,
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('Dataset Versions panel', () => {
  it('renders the version tree and highlights the selected branch', async () => {
    mockFetch({
      ...baseHandlers,
      'GET /projects/p1/dataset_versions': () => [V1, V2, V3, V4],
      'GET /dataset-versions/v3/preview': () => previewBody(1),
    })
    renderWithProviders(<App />, { route: '/projects/p1/dataset?v=v3' })

    const tree = await screen.findByRole('button', { name: /v3/ })
    expect(tree).toHaveAttribute('aria-current', 'true')
    expect(screen.getByRole('button', { name: /v1/ })).toHaveAttribute('data-branch', 'ancestor')
    expect(screen.getByRole('button', { name: /v2/ })).toHaveAttribute('data-branch', 'ancestor')
    // the sibling branch stays unhighlighted
    expect(screen.getByRole('button', { name: /v4/ })).not.toHaveAttribute('data-branch')
    // provenance breakdown present (Jev segment on the labeled version)
    expect(tree).toHaveTextContent('60')
  })

  it('pages the preview table and shows column types', async () => {
    const user = userEvent.setup()
    const { calls } = mockFetch({
      ...baseHandlers,
      'GET /projects/p1/dataset_versions': () => [V1, V2, V3],
      'GET /dataset-versions/v3/preview': () => previewBody(1),
    })
    renderWithProviders(<App />, { route: '/projects/p1/dataset?v=v3' })

    expect(await screen.findByText('age')).toBeInTheDocument()
    expect(screen.getByText('integer')).toBeInTheDocument()
    expect(screen.getByText(/rows 1–25 of 60/)).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Next' }))
    await waitFor(() =>
      expect(calls.some(([m, p]) => m === 'GET' && p.includes('preview') && p.includes('page=2'))).toBe(
        true,
      ),
    )
  })

  it('offers CSV and Parquet downloads for the selected version', async () => {
    mockFetch({
      ...baseHandlers,
      'GET /projects/p1/dataset_versions': () => [V1, V2, V3],
      'GET /dataset-versions/v3/preview': () => previewBody(1),
    })
    renderWithProviders(<App />, { route: '/projects/p1/dataset?v=v3' })
    const csv = await screen.findByRole('link', { name: /Download CSV/ })
    expect(csv).toHaveAttribute('href', expect.stringContaining('format=csv'))
    expect(screen.getByRole('link', { name: /Parquet/ })).toHaveAttribute(
      'href',
      expect.stringContaining('format=parquet'),
    )
  })

  it('uploading a file creates a version, selects it, and the tree updates', async () => {
    let uploaded = false
    mockFetch({
      ...baseHandlers,
      'GET /projects/p1/dataset_versions': () =>
        uploaded ? [V1, V2, V3, V4, version('v5', 5, 'uploaded', null, { uploaded: 9 })] : [V1, V2, V3, V4],
      'POST /projects/p1/upload': () => {
        uploaded = true
        return version('v5', 5, 'uploaded', null, { uploaded: 9 })
      },
      'GET /dataset-versions/v5/preview': () => ({ ...previewBody(1), version_id: 'v5', total_rows: 9 }),
    })
    renderWithProviders(<App />, { route: '/projects/p1/dataset?v=v1' })
    await screen.findByRole('button', { name: /v4/ })

    const input = screen.getByLabelText('Upload data file')
    const file = new File(['age,plan\n31,basic'], 'people.csv', { type: 'text/csv' })
    fireEvent.change(input, { target: { files: [file] } })

    // the new root appears in the tree and becomes the selected version
    expect(await screen.findByRole('button', { name: /v5/ })).toBeInTheDocument()
    await waitFor(() => expect(uploaded).toBe(true))
    const previewHeading = await screen.findByText('Version 5')
    expect(previewHeading).toBeInTheDocument()
  })

  it('shows upload errors inline', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const path = String(input).replace(/^\/api/, '')
        const ok = (body: unknown) =>
          new Response(JSON.stringify(body), {
            status: 200,
            headers: { 'content-type': 'application/json' },
          })
        if ((init?.method ?? 'GET') === 'POST') {
          return new Response(JSON.stringify({ detail: 'unsupported file type .exe' }), {
            status: 415,
            headers: { 'content-type': 'application/json' },
          })
        }
        if (path.includes('dataset_versions')) return ok([V1])
        if (path.includes('preview')) return ok({ ...previewBody(1), version_id: 'v1' })
        if (path === '/projects/p1') return ok(project)
        return ok([project])
      }),
    )
    renderWithProviders(<App />, { route: '/projects/p1/dataset?v=v1' })
    await screen.findByRole('button', { name: /v1/ })
    const input = screen.getByLabelText('Upload data file')
    fireEvent.change(input, { target: { files: [new File(['MZ'], 'virus.exe')] } })
    expect(await screen.findByRole('alert')).toHaveTextContent('unsupported file type')
  })

  it('ends the step with the Checks panel, gated on the selected version', async () => {
    const user = userEvent.setup()
    const pii = {
      id: 'c1',
      kind: 'pii_found',
      severity: 'warning' as const,
      message: 'Possible email addresses were found in 12 rows.',
      subject_type: 'dataset_version' as const,
      subject_id: 'v3',
      details: { columns: ['email'] },
      acknowledged: false,
      acknowledged_at: null,
      note: null,
    }
    const acked = new Set<string>()
    const { calls } = mockFetch({
      ...baseHandlers,
      'GET /projects/p1/dataset_versions': () => [V1, V2, V3],
      'GET /dataset-versions/v3/preview': () => previewBody(1),
      'GET /checks': () => {
        const check = { ...pii, acknowledged: acked.has('c1'), acknowledged_at: '2024-01-01T00:00:00Z' }
        return { checks: [check], unacknowledged_warnings: acked.has('c1') ? 0 : 1 }
      },
      'POST /checks/c1/acknowledge': () => {
        acked.add('c1')
        return { ...pii, acknowledged: true, acknowledged_at: '2024-01-01T00:00:00Z' }
      },
    })
    renderWithProviders(<App />, { route: '/projects/p1/dataset?v=v3' })

    expect(await screen.findByText(pii.message)).toBeInTheDocument()
    // the gate is on the version the user is looking at
    expect(
      calls.some(([, p]) => p.startsWith('/checks?subject_type=dataset_version&subject_id=v3')),
    ).toBe(true)

    const gate = screen.getByRole('button', { name: 'Go to Generate' })
    expect(gate).toBeDisabled()
    await user.click(screen.getByRole('button', { name: 'Acknowledge' }))
    await user.click(screen.getByRole('button', { name: 'Record Acknowledgement' }))
    await waitFor(() => expect(gate).toBeEnabled())
  })

  it('empty tree nudges toward upload', async () => {
    mockFetch({
      ...baseHandlers,
      'GET /projects/p1/dataset_versions': () => [],
    })
    renderWithProviders(<App />, { route: '/projects/p1/dataset' })
    expect(await screen.findByText(/No Dataset Versions yet/)).toBeInTheDocument()
  })
})
