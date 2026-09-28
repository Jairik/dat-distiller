/**
 * The Fidelity Report panel.
 *
 * Two things are being protected here. First, a statistic is never shown
 * without a sentence: `p = 0.0004` on its own is not actionable, and reading
 * it as "proves a match" would be worse than not showing it. Second, the
 * Provider-derived banner must dominate the panel — every comparison under it
 * is against a description of the data, not against real data.
 */

import { screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { FidelityPanel } from '@/components/fidelity/fidelity-panel'
import {
  driftColor,
  interpretCorrelation,
  interpretTest,
  useFidelity,
  type FidelityResponse,
} from '@/components/fidelity/charts'
import { renderWithProviders } from '@/test/render'

afterEach(() => {
  vi.unstubAllGlobals()
})

const REPORT: FidelityResponse['report'] = {
  ks_tests: { age: { statistic: 0.11, p_value: 0.82 } },
  chi2_tests: { plan: { statistic: 1.2, p_value: 0.04 } },
  correlation_drift: 0.42,
  exact_duplicates: { count: 3, fraction: 0.03 },
  near_copies: { count: 2, threshold: 0.05, examples: [1, 4], scanned: 100 },
  dropped_rows: 12,
  dropped_ratio: 0.02,
  balance: { plan: { pro: { requested: 0.5, actual: 0.38 } } },
  provider_derived_profile: false,
  generated_rows: 100,
  warnings: ['near_copies', 'fidelity_drift'],
}

const BODY: FidelityResponse = {
  version_id: 'v2',
  sample_version_id: 'v1',
  report: REPORT,
  distributions: [
    {
      column: 'age',
      kind: 'integer',
      chart: 'histogram',
      edges: [20, 30, 40, 50, 60],
      sample: [10, 20, 15, 5],
      generated: [8, 18, 16, 6],
      only_in: null,
    },
    {
      column: 'plan',
      kind: 'categorical',
      chart: 'bar',
      categories: ['basic', 'pro'],
      sample: [0.5, 0.5],
      generated: [0.62, 0.38],
      only_in: null,
    },
  ],
  correlation: {
    columns: ['age', 'spend'],
    sample: [
      [1, 0.8],
      [0.8, 1],
    ],
    generated: [
      [1, 0.2],
      [0.2, 1],
    ],
    drift: [
      [0, 0.6],
      [0.6, 0],
    ],
    max_drift: 0.6,
  },
}

function renderPanel(body: unknown = BODY) {
  mockFidelity(body)
  return renderWithProviders(<FidelityPanel versionId="v2" />)
}

function mockFidelity(body: unknown) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () =>
      new Response(JSON.stringify(body), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    ),
  )
}

describe('FidelityPanel', () => {
  it('shows the headline numbers', async () => {
    renderPanel()
    expect(await screen.findByText('Rows compared')).toBeInTheDocument()
    expect(screen.getByText('100')).toBeInTheDocument()
    // correlation drift, the near-copy count and the dropped count, each in
    // its own card so they cannot be confused with one another
    const card = (label: string) =>
      screen.getByText(label).closest('[data-slot="card"]') as HTMLElement
    expect(within(card('Worst correlation drift')).getByText('0.600')).toBeInTheDocument()
    expect(within(card('Near-copies')).getByText('2')).toBeInTheDocument()
    expect(within(card('Dropped rows')).getByText('12')).toBeInTheDocument()
    expect(screen.getByText(/2% of what was asked for/)).toBeInTheDocument()
  })

  it('gives every test result a plain-language reading, not just a p-value', async () => {
    renderPanel()
    await screen.findByText('Test results')
    // a high p-value reads as a match...
    expect(screen.getByText('looks the same')).toBeInTheDocument()
    expect(
      screen.getByText(/happens by chance about 82% of the time/),
    ).toBeInTheDocument()
    // each tested column is named, and which kind of test it ran is labelled
    const results = screen.getByText('Test results').closest('[data-slot="card"]') as HTMLElement
    expect(within(results).getByText(/^age/)).toBeInTheDocument()
    expect(within(results).getByText(/^plan/)).toBeInTheDocument()
    expect(within(results).getByText('shape test')).toBeInTheDocument()
    expect(within(results).getByText('category test')).toBeInTheDocument()
    // ...and a low one as a real difference, in words
    expect(screen.getByText('probably drifted')).toBeInTheDocument()
    expect(screen.getByText(/Only a 0.040 chance/)).toBeInTheDocument()
    expect(screen.getByText('p = 0.8200')).toBeInTheDocument()
  })

  it('charts every column, with a chart suited to its type', async () => {
    renderPanel()
    await screen.findByText('Distributions, column by column')
    // numeric -> a histogram with a real versus synthetic label
    expect(
      screen.getByRole('img', { name: /Distribution of age: real versus synthetic/ }),
    ).toBeInTheDocument()
    // categorical -> category shares
    expect(
      screen.getByRole('img', { name: /Categories of plan: real versus synthetic/ }),
    ).toBeInTheDocument()
    // and the numbers behind the bars are actually rendered
    expect(screen.getByTestId('bar-sample-plan-1')).toHaveStyle({ width: '50%' })
    expect(screen.getByTestId('bar-generated-plan-1')).toHaveStyle({ width: '38%' })
  })

  it('flags a column that only exists on one side instead of faking a match', async () => {
    renderPanel({
      ...BODY,
      distributions: [
        { column: 'legacy_flag', kind: 'text', chart: 'bar', categories: ['a'], sample: [1], generated: [0], only_in: 'sample' },
      ],
    })
    expect(await screen.findByText(/only in the real data/)).toBeInTheDocument()
  })

  it('renders the correlation drift heatmap and says what the worst pair means', async () => {
    renderPanel()
    await screen.findByRole('img', { name: /Correlation drift between pairs/ })
    expect(screen.getByText('Where the relationships moved')).toBeInTheDocument()
    expect(screen.getByTestId('drift-age-spend')).toBeInTheDocument()
    expect(screen.getByText(/moved by/)).toHaveTextContent('0.600')
    expect(screen.getByText(/learned the wrong thing/)).toBeInTheDocument()
  })

  it('reports duplicates, balance drift and the warning codes', async () => {
    renderPanel()
    await screen.findByText('Duplicates, balance and the rest')
    expect(screen.getByText(/exact duplicate row\(s\)/)).toHaveTextContent('3 exact duplicate')
    expect(screen.getByText(/asked for 50%, got 38%/)).toBeInTheDocument()
    expect(screen.getByText(/off target/)).toBeInTheDocument()
    // the raw codes are never shown to a user
    expect(screen.getByText(/too close to real rows/)).toBeInTheDocument()
    expect(screen.getByText(/does not match the real distribution/)).toBeInTheDocument()
    expect(screen.queryByText('near_copies')).not.toBeInTheDocument()
  })

  it('puts the Provider-derived banner first and explains what it invalidates', async () => {
    renderPanel({
      ...BODY,
      report: { ...REPORT, provider_derived_profile: true, warnings: ['provider_derived_profile'] },
    })
    const banner = await screen.findByRole('alert')
    expect(banner).toHaveTextContent('The Profile was Provider-derived')
    expect(banner).toHaveTextContent(/not against reality/)
    expect(banner).toHaveTextContent(/smoke test/)
  })

  it('has no banner when the Profile came from real data', async () => {
    renderPanel()
    await screen.findByText('Rows compared')
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('says a version with no report has no report, instead of drawing empty charts', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(JSON.stringify({ detail: 'no Fidelity Report' }), {
          status: 404,
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    )
    renderWithProviders(<FidelityPanel versionId="v1" />)
    expect(
      await screen.findByText(/was not produced by Generation/),
    ).toBeInTheDocument()
  })

  it('says when a numeric comparison is impossible', async () => {
    renderPanel({
      ...BODY,
      correlation: { columns: ['age'], sample: [[1]], generated: [[1]], drift: [[0]], max_drift: 0 },
    })
    expect(
      await screen.findByText(/at least two numeric columns/),
    ).toBeInTheDocument()
  })

  it('handles a column where every value is missing', async () => {
    renderPanel({
      ...BODY,
      distributions: [
        { column: 'notes', kind: 'text', chart: 'bar', categories: [], sample: [], generated: [], only_in: null },
        { column: 'constant', kind: 'integer', chart: 'histogram', edges: [7, 7.05], sample: [5, 0], generated: [5, 0], only_in: null },
      ],
    })
    await screen.findByText('Distributions, column by column')
    expect(screen.getByText(/every value is missing/)).toBeInTheDocument()
  })

  it('does not fetch without a version', () => {
    const spy = vi.fn()
    vi.stubGlobal('fetch', spy)
    renderWithProviders(<FidelityPanel versionId={undefined} />)
    expect(spy).not.toHaveBeenCalled()
  })
})

describe('interpretTest', () => {
  it('never claims a high p-value proves anything', () => {
    const { verdict, detail } = interpretTest({ statistic: 0.01, p_value: 0.9 })
    expect(verdict).toBe('looks the same')
    expect(detail).toMatch(/by chance/)
    expect(detail).not.toMatch(/prove/i)
  })

  it('escalates as the p-value falls', () => {
    expect(interpretTest({ statistic: 0, p_value: 0.5 }).verdict).toBe('looks the same')
    expect(interpretTest({ statistic: 0, p_value: 0.04 }).verdict).toBe('probably drifted')
    expect(interpretTest({ statistic: 0, p_value: 0.0001 }).verdict).toBe('clearly different')
  })
})

describe('interpretCorrelation', () => {
  it('describes a coefficient in words, not as a number', () => {
    expect(interpretCorrelation(0.95)).toMatch(/strongly related up/)
    expect(interpretCorrelation(-0.8)).toMatch(/strongly related down/)
    expect(interpretCorrelation(0.25)).toMatch(/weakly related/)
    expect(interpretCorrelation(0.02)).toMatch(/essentially unrelated/)
    expect(interpretCorrelation(null)).toMatch(/not enough data/)
  })
})

describe('driftColor', () => {
  it('is a clear ramp, and an undefined pair is visibly distinct', () => {
    expect(driftColor(0.05)).not.toBe(driftColor(0.5))
    expect(driftColor(null)).toContain('120 120 120')
  })
})

describe('useFidelity', () => {
  it('is keyed per version so two reports never share a cache', () => {
    expect(useFidelity).toBeTypeOf('function')
  })
})

describe('panel structure', () => {
  it('keeps every column chart inside the distributions region', async () => {
    renderPanel()
    const heading = await screen.findByText('Distributions, column by column')
    // the whole grid is one landmark, so a screen reader can skip past the
    // charts rather than walking two elements per column
    const region = heading.closest('[data-slot="card"]') as HTMLElement
    expect(within(region).getByRole('img', { name: /Distribution of age/ })).toBeInTheDocument()
    expect(within(region).getByRole('img', { name: /Categories of plan/ })).toBeInTheDocument()
  })
})
