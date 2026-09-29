/**
 * The Fairness Report view.
 *
 * The two acceptance criteria are both about *readability and consequence*:
 *
 * - **Groups and gaps readable at a glance.** Tested by checking the bars carry
 *   accessible names *and* the numbers are in a table beside them, and that the
 *   worst gap is described in a sentence naming both ends of it — a reader who
 *   sees "TPR gap 0.31" has learned nothing about which group was treated worse.
 * - **Flagged gaps become Checks that need Acknowledgement.** Tested by running a
 *   report that flags, then driving the real Checks panel to acknowledgement and
 *   asserting the gate opens. A report that raises a Check nobody can acknowledge
 *   is a dead end.
 *
 * The case that is easiest to get wrong and is therefore tested hardest: a group
 * the report could not measure must appear, greyed, with its reason. Dropping it
 * turns a three-group report into a two-group one and reads as "these two are
 * treated equally", which is not what the data says.
 */

import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import App from '@/App'

/**
 * Recharts stands in for itself, and renders the data it was handed.
 *
 * The panel's own test asserted only on the `role="img"` wrapper, so hard-coding
 * every bar's value to 0 *and deleting the `<Bar>` element entirely* left all 20
 * tests in this file passing — the chart was not under test at all. Recharts
 * measures nothing in jsdom (`ResponsiveContainer` has zero size), so stubbing
 * it is the only honest way to see what the chart was given.
 */
vi.mock('recharts', () => {
  const Box = ({ children }: { children?: React.ReactNode }) => <div>{children}</div>
  return {
    ResponsiveContainer: Box,
    BarChart: ({
      data,
      children,
    }: {
      data?: Array<Record<string, unknown>>
      children?: React.ReactNode
    }) => (
      <div data-testid="bar-chart" data-rows={JSON.stringify(data ?? [])}>
        {children}
      </div>
    ),
    Bar: ({ dataKey }: { dataKey?: string }) => <div data-testid="bar" data-key={dataKey} />,
    XAxis: (props: Record<string, unknown>) => (
      <div data-testid="x-axis" data-domain={JSON.stringify(props.domain)} />
    ),
    YAxis: () => <div data-testid="y-axis" />,
    CartesianGrid: () => <div data-testid="grid" />,
    Tooltip: () => null,
    ReferenceLine: () => <div data-testid="reference-line" />,
  }
})
import { explainGap, headlineGap, type FairnessReport } from '@/lib/fairness'
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
  name: 'Fair Lab',
  created_at: '2024-01-01T00:00:00Z',
  version_count: 1,
  latest_version_id: 'v1',
}
const V1 = {
  id: 'v1',
  project_id: 'p1',
  parent_id: null,
  number: 1,
  origin: 'labeled',
  row_count: 400,
  columns: [{ name: 'group', kind: 'categorical' }],
  provenance_summary: { jev: 400 },
  seed: 3,
  meta: {},
  created_at: '2024-01-01T00:00:00Z',
}

const PLAN = {
  version_id: 'v1',
  target: 'is_churn',
  task_type: 'classification',
  label_family: 'is_churn',
  seed: 1,
  rows: 400,
  kept_rows: 360,
  train_rows: 288,
  test_rows: 72,
  n_source_features: 2,
  models: [],
  model_names: ['logistic_regression'],
  unavailable_models: [],
  primary_metric: 'f1_macro',
  primary_metric_label: 'F1 (macro)',
  primary_metric_higher_is_better: true,
  metrics_available: ['accuracy', 'f1_macro'],
  tuning: { enabled: false, strategy: 'random', n_iter: 10, cv: 3, tunable: {} },
  hyperparameters: {},
}

const ENTRY = {
  rank: 1,
  model: 'logistic_regression',
  label: 'Logistic Regression',
  library: 'sklearn',
  task_type: 'classification',
  status: 'ok' as const,
  classes: ['bool:False', 'bool:True'],
  hyperparameters: {},
  fit_seconds: 0.1,
  metrics: { f1_macro: { value: 0.9, reason: null } },
  primary_metric: 'f1_macro',
  primary: { value: 0.9, reason: null },
}

const RUN = {
  training_run_id: 'j1',
  job_id: 'j1',
  project_id: 'p1',
  version_id: 'v1',
  status: 'completed',
  progress: {},
  created_at: '2024-01-02T00:00:00Z',
  finished_at: '2024-01-02T00:01:00Z',
  error: null,
  target: 'is_churn',
  task_type: 'classification',
  seed: 1,
  primary_metric: 'f1_macro',
  primary_metric_higher_is_better: true,
  leaderboard: [ENTRY],
  warnings: [],
  setup: {
    target: 'is_churn',
    sensitive_attribute: 'group',
    feature_columns: ['age', 'spend'],
    excluded_columns: { group: 'sensitive_attribute', is_churn__confidence: 'label_sibling_exclusion' },
  },
}

function group(
  name: string,
  metrics: Record<string, { value: number | null; reason: string | null }>,
  over: Partial<FairnessReport['groups'][number]> = {},
) {
  return {
    group: name,
    n_version: 100,
    n_total: 100,
    n_train: 80,
    n_test: 40,
    n_actual_positive: 20,
    n_actual_negative: 20,
    n_predicted_positive: 20,
    class_counts: { '1': 20, '0': 20 },
    measured: true,
    reason: null,
    metrics,
    ...over,
  }
}

function gap(over: Partial<FairnessReport['gaps'][number]> = {}) {
  return {
    name: 'accuracy',
    label: 'Accuracy difference',
    metric: 'accuracy',
    unit: '',
    description: 'how far the groups differ in accuracy',
    threshold: 0.1,
    groups_total: 2,
    groups_compared: 2,
    groups_unmeasured: [],
    value: 0.18,
    reason: null,
    comparable: true,
    exceeds: true,
    is_widest: false,
    ...over,
  }
}

const REPORT: FairnessReport = {
  training_run_id: 'j1',
  project_id: 'p1',
  version_id: 'v1',
  target: 'is_churn',
  task_type: 'classification',
  model: 'logistic_regression',
  model_label: 'Logistic Regression',
  seed: 1,
  refit: true,
  sensitive_attribute: 'group',
  declared_sensitive_attribute: 'group',
  sensitive_attribute_excluded_from_features: true,
  positive_label: null,
  positive_class: 'bool:True',
  classes: ['bool:False', 'bool:True'],
  threshold: 0.1,
  threshold_source: 'request',
  split: { train_rows: 288, test_rows: 72, reused_stored_split: true },
  overall: { accuracy: { value: 0.88, reason: null } },
  n_groups: 3,
  n_groups_measured: 2,
  groups: [
    group('str:a', { accuracy: { value: 0.95, reason: null }, tpr: { value: 0.9, reason: null } }),
    group('str:b', { accuracy: { value: 0.77, reason: null }, tpr: { value: 0.2, reason: null } }),
    group('str:c', { accuracy: { value: null, reason: 'no rows in the held-out split' } }, {
      measured: false,
      reason: "group 'str:c' has no rows among the rows this Training Run kept; all 100 of its row(s) in this Dataset Version were dropped before the split",
      n_test: 0,
    }),
  ],
  unmeasured_groups: [
    {
      group: 'str:c',
      reason: "group 'str:c' has no rows among the rows this Training Run kept; all 100 of its row(s) in this Dataset Version were dropped before the split",
      n_test: 0,
    },
  ],
  gaps: [
    gap({ is_widest: true, groups_unmeasured: ['str:c'] }),
    gap({
      name: 'tpr',
      label: 'True positive rate difference',
      metric: 'tpr',
      value: 0.7,
      exceeds: true,
    }),
    gap({ name: 'demographic_parity', label: 'Demographic parity difference', metric: 'selection_rate', value: 0.03, exceeds: false, is_widest: false }),
    gap({
      name: 'fpr',
      label: 'False positive rate difference',
      metric: 'fpr',
      value: 0.02,
      exceeds: false,
    }),
  ],
  gaps_exceeding_threshold: ['tpr', 'accuracy'],
  largest_gap: gap({ is_widest: true }),
  largest_comparable_gap: gap({ is_widest: true }),
  gaps_within_threshold: ['fpr', 'demographic_parity'],
  gaps_not_compared: [],
  notes: ['The Model was refitted for this report.'],
  checks_raised: [
    {
      id: 'c1',
      kind: 'fairness_gap',
      severity: 'warning',
      message: "Model 'Logistic Regression' by 'group': true positive rate difference 0.700 exceeds the threshold of 0.1",
      subject_type: 'training_run',
      subject_id: 'j1',
      details: { gap: 'tpr' },
      acknowledged: false,
      acknowledged_at: null,
      note: null,
    },
  ],
  check_step: 'fairness',
  unacknowledged_warnings: [],
}

const baseHandlers = {
  'GET /projects': () => [project],
  'GET /projects/p1': () => project,
  'GET /projects/p1/dataset_versions': () => [V1],
  'GET /dataset-versions/v1/preview': () => ({
    version_id: 'v1',
    columns: [
      { name: 'age', kind: 'integer' },
      { name: 'is_churn', kind: 'bool' },
    ],
    page: 0,
    page_size: 1,
    total_rows: 400,
    rows: [[31, true]],
  }),
  'GET /train/models': () => ({
    models: [
      {
        name: 'logistic_regression',
        label: 'Logistic Regression',
        library: 'sklearn',
        extra: 'sklearn',
        task_types: ['classification'],
        supports_proba: true,
        notes: '',
        available: true,
      },
    ],
    extras: {},
    library_versions: {},
  }),
  'GET /train/metrics': () => ({ task_types: {} }),
  'GET /train/runs': () => ({ project_id: 'p1', count: 0, runs: [] }),
  'POST /train/plan': () => PLAN,
  'POST /train/run': () => ({ id: 'j1' }),
  'GET /train/runs/j1': () => ({ ...RUN, status: 'running' }),
  'POST /train/runs/j1/fairness': () => REPORT,
}

/** A regression report: `mae` is an error in the Target's units, not a rate. */
const REGRESSION_REPORT: FairnessReport = {
      ...REPORT,
      task_type: 'regression',
      target: 'spend_target',
      positive_class: null,
      groups: [
        group('str:a', { mae: { value: 1.2, reason: null }, mae_relative: { value: 0.1, reason: null } }),
        group('str:b', { mae: { value: 3.4, reason: null }, mae_relative: { value: 0.28, reason: null } }),
      ],
      n_groups: 2,
      n_groups_measured: 2,
      unmeasured_groups: [],
      gaps: [
        gap({
          name: 'mae',
          label: 'Mean absolute error difference',
          metric: 'mae',
          unit: 'Target units',
          value: 2.2,
          comparable: false,
          exceeds: null,
        }),
        gap({
          name: 'mae_relative',
          label: 'Mean absolute error difference (relative)',
          metric: 'mae_relative',
          value: 0.18,
          exceeds: true,
        }),
      ],
      gaps_exceeding_threshold: ['mae_relative'],
    }

async function showReport(
  user: ReturnType<typeof userEvent.setup>,
  handlers: Record<string, unknown> = {},
) {
  mockFetch({ ...baseHandlers, ...handlers })
  renderWithProviders(<App />, { route: '/projects/p1/train?version=v1' })
  await user.click(await screen.findByLabelText('is_churn', { selector: '#target-is_churn' }))
  await user.click(await screen.findByRole('button', { name: 'See the plan' }))
  await screen.findByTestId('train-plan')
  await user.click(screen.getByRole('button', { name: 'Train' }))
  await screen.findByText('Training Run')
  await emitJobEvent('j1', 'completed', {
    status: 'completed',
    progress: {},
    result: RUN,
    error: null,
  })
  await user.click(await screen.findByRole('button', { name: 'Fairness Report' }))
  await user.click(await screen.findByRole('button', { name: 'Measure this group' }))
  return screen.findByTestId('fairness-headline')
}

describe('Fairness Report — asking for it', () => {
  it('computes nothing until the user asks', async () => {
    const user = userEvent.setup()
    const { calls } = mockFetch(baseHandlers)
    renderWithProviders(<App />, { route: '/projects/p1/train?version=v1' })
    await user.click(await screen.findByLabelText('is_churn', { selector: '#target-is_churn' }))
    await user.click(await screen.findByRole('button', { name: 'See the plan' }))
    await screen.findByTestId('train-plan')
    await user.click(screen.getByRole('button', { name: 'Train' }))
    await screen.findByText('Training Run')
    await emitJobEvent('j1', 'completed', {
      status: 'completed',
      progress: {},
      result: RUN,
      error: null,
    })
    await screen.findByRole('button', { name: 'Fairness Report' })
    // even opening the panel asks for nothing
    await user.click(screen.getByRole('button', { name: 'Fairness Report' }))
    expect(calls.some(([, p]) => p.includes('/fairness'))).toBe(false)
  })

  it('pre-selects the run’s own Sensitive Attribute', async () => {
    const user = userEvent.setup()
    mockFetch(baseHandlers)
    renderWithProviders(<App />, { route: '/projects/p1/train?version=v1' })
    await user.click(await screen.findByLabelText('is_churn', { selector: '#target-is_churn' }))
    await user.click(await screen.findByRole('button', { name: 'See the plan' }))
    await screen.findByTestId('train-plan')
    await user.click(screen.getByRole('button', { name: 'Train' }))
    await screen.findByText('Training Run')
    await emitJobEvent('j1', 'completed', { status: 'completed', progress: {}, result: RUN, error: null })
    await user.click(await screen.findByRole('button', { name: 'Fairness Report' }))
    const picker = await screen.findByLabelText('Report on this Sensitive Attribute')
    expect(picker).toHaveValue('group')
  })

  it('sends the attribute, the Model and the threshold', async () => {
    const user = userEvent.setup()
    const { bodies } = mockFetch(baseHandlers)
    renderWithProviders(<App />, { route: '/projects/p1/train?version=v1' })
    await user.click(await screen.findByLabelText('is_churn', { selector: '#target-is_churn' }))
    await user.click(await screen.findByRole('button', { name: 'See the plan' }))
    await screen.findByTestId('train-plan')
    await user.click(screen.getByRole('button', { name: 'Train' }))
    await screen.findByText('Training Run')
    await emitJobEvent('j1', 'completed', { status: 'completed', progress: {}, result: RUN, error: null })
    await user.click(await screen.findByRole('button', { name: 'Fairness Report' }))
    await user.type(await screen.findByLabelText('Gap threshold (optional)'), '0.25')
    await user.click(screen.getByRole('button', { name: 'Measure this group' }))
    await waitFor(() =>
      expect(bodies).toContainEqual(
        expect.objectContaining({ sensitive_attribute: 'group', gap_threshold: 0.25 }),
      ),
    )
  })

  it('reports a request the backend refused', async () => {
    const user = userEvent.setup()
    mockFetch({
      ...baseHandlers,
      'POST /train/runs/j1/fairness': () => {
        throw new Error('the Sensitive Attribute cannot also be the Target')
      },
    })
    renderWithProviders(<App />, { route: '/projects/p1/train?version=v1' })
    await user.click(await screen.findByLabelText('is_churn', { selector: '#target-is_churn' }))
    await user.click(await screen.findByRole('button', { name: 'See the plan' }))
    await screen.findByTestId('train-plan')
    await user.click(screen.getByRole('button', { name: 'Train' }))
    await screen.findByText('Training Run')
    await emitJobEvent('j1', 'completed', { status: 'completed', progress: {}, result: RUN, error: null })
    await user.click(await screen.findByRole('button', { name: 'Fairness Report' }))
    await user.click(screen.getByRole('button', { name: 'Measure this group' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('cannot also be the Target')
  })
})

describe('Fairness Report — readable at a glance', () => {
  it('draws one bar per group per metric, with the numbers beside them', async () => {
    const user = userEvent.setup()
    await showReport(user)
    const bars = await screen.findByTestId('fairness-bars')
    expect(within(bars).getByRole('img', { name: 'Accuracy by group' })).toBeInTheDocument()
    const table = within(bars).getByTestId('table-accuracy')
    const rows = within(table).getAllByRole('row').slice(1)
    expect(rows.map((r) => r.textContent)).toEqual([
      'str:a400.9500',
      'str:b400.7700',
    ])
  })

  it('charts the values it was given, and leaves out the groups it has none for', async () => {
    const user = userEvent.setup()
    await showReport(user)
    const bars = await screen.findByTestId('fairness-bars')
    const charts = within(bars).getAllByTestId('bar-chart')
    expect(charts.length).toBeGreaterThan(0)

    for (const chart of charts) {
      const rows = JSON.parse(chart.getAttribute('data-rows') ?? '[]') as Array<{
        group: string
        value: number
      }>
      // Every plotted value is a real measurement, never a stand-in zero.
      expect(rows.length).toBeGreaterThan(0)
      for (const row of rows) {
        expect(Number.isFinite(row.value)).toBe(true)
      }
    }

    // The bars are actually rendered, not just prepared.
    expect(within(bars).getAllByTestId('bar').length).toBeGreaterThan(0)
  })

  it('leaves a group with no value for a metric out of that chart, not in at zero', async () => {
    // The numbers table prints an em dash for exactly these groups, so `?? 0`
    // made the chart and the table disagree about the same measurement — and a
    // zero-length bar says "this group scored nothing", which is not the same
    // claim as "there was nothing to measure".
    const user = userEvent.setup()
    const partial: FairnessReport = {
      ...REPORT,
      groups: [
        group('str:a', { accuracy: { value: 0.95, reason: null } }),
        group('str:b', { accuracy: { value: null, reason: 'this group has one class only' } }),
        group('str:c', { accuracy: { value: 0.4, reason: null } }),
      ],
    }
    await showReport(user, { 'POST /train/runs/j1/fairness': () => partial })
    const bars = await screen.findByTestId('fairness-bars')
    const accuracy = within(bars).getByTestId('bars-accuracy')
    const rows = JSON.parse(
      accuracy.querySelector('[data-testid="bar-chart"]')!.getAttribute('data-rows')!,
    ) as Array<{ group: string; value: number }>

    expect(rows.map((r) => r.group)).toEqual(['str:a', 'str:c'])
    expect(rows.every((r) => r.value > 0)).toBe(true)
    // The table still accounts for all three, with the gap marked.
    const table = within(bars).getByTestId('table-accuracy')
    expect(within(table).getAllByRole('row').slice(1)).toHaveLength(3)
  })

  it('scales the axis to the metric, so a regression error is not clipped to full width', async () => {
    // `domain={[0, 1]}` is right for a rate and wrong for an error: an MAE of
    // 1.2 and an MAE of 3.4 both clamp to a full-width bar, so the headline
    // chart of a regression report was a row of identical bars.
    const user = userEvent.setup()
    await showReport(user, { 'POST /train/runs/j1/fairness': () => REGRESSION_REPORT })
    const bars = await screen.findByTestId('fairness-bars')
    const mae = within(bars).getByTestId('bars-mae')
    const domain = JSON.parse(mae.querySelector('[data-testid="x-axis"]')!.getAttribute('data-domain')!)
    const [low, high] = domain as [number, number]
    expect(low).toBe(0)
    expect(high).toBeGreaterThanOrEqual(3.4)

    // And the wider error is drawn wider, which is the whole point of the chart.
    const chart = mae.querySelector('[data-testid="bar-chart"]')!
    const rows = JSON.parse(chart.getAttribute('data-rows')!) as Array<{ value: number }>
    expect(rows[0].value).not.toBe(rows[1].value)
  })

  it('says which group was treated worse, in a sentence', async () => {
    const user = userEvent.setup()
    const headline = await showReport(user)
    // a reader who sees "0.180" has learned nothing about which group was worse
    expect(headline).toHaveTextContent('str:a')
    expect(headline).toHaveTextContent('str:b')
    expect(headline).toHaveTextContent('over the 0.1 threshold')
  })

  it('lists every gap, flagged or not', async () => {
    const user = userEvent.setup()
    await showReport(user)
    const gaps = await screen.findByTestId('fairness-gaps')
    for (const name of ['accuracy', 'tpr', 'demographic_parity', 'fpr']) {
      expect(within(gaps).getByTestId(`gap-${name}`)).toBeInTheDocument()
    }
    expect(within(gaps).getAllByText('over the threshold')).toHaveLength(2)
    // and marks the widest one
    expect(within(gaps).getAllByText('widest').length).toBeGreaterThan(0)
  })

  it('shows a group it could not measure, with the reason — never drops it', async () => {
    const user = userEvent.setup()
    await showReport(user)
    const unmeasured = await screen.findByTestId('unmeasured-groups')
    const row = within(unmeasured).getByTestId('unmeasured-str:c')
    expect(row).toHaveTextContent('dropped before the split')
    // and the headline says how many could be measured, out of how many exist
    const headline = screen.getByTestId('fairness-headline')
    expect(headline).toHaveTextContent('2 of 3 group(s) could be measured')
  })

  it('says a gap in the Target’s own units cannot be judged by a rate threshold', async () => {
    const user = userEvent.setup()
    await showReport(user, { 'POST /train/runs/j1/fairness': () => REGRESSION_REPORT })

    const gaps = await screen.findByTestId('fairness-gaps')
    const mae = within(gaps).getByTestId('gap-mae')
    expect(mae).toHaveTextContent('2.2000 Target units')
    expect(mae).toHaveTextContent('not judged')
    // and the relative one is the one that counts
    expect(within(gaps).getByTestId('gap-mae_relative')).toHaveTextContent('over the threshold')
  })

  it('flags that the Model could see the attribute when it could', async () => {
    const user = userEvent.setup()
    await showReport(user, {
      'POST /train/runs/j1/fairness': () => ({
        ...REPORT,
        sensitive_attribute_excluded_from_features: false,
      }),
    })
    const headline = await screen.findByTestId('fairness-headline')
    expect(headline).toHaveTextContent('the Model could see this attribute')
  })

  it('states the caveats, including that the Model was refit', async () => {
    const user = userEvent.setup()
    await showReport(user)
    const notes = await screen.findByTestId('fairness-notes')
    expect(notes).toHaveTextContent('refitted for this report')
    expect(notes).toHaveTextContent('one held-out split')
  })
})

describe('AC2 — flagged gaps become Checks needing Acknowledgement', () => {
  it('says how many Checks were raised and why', async () => {
    const user = userEvent.setup()
    await showReport(user)
    const summary = await screen.findByTestId('checks-raised-summary')
    expect(summary).toHaveTextContent('2 gap(s) over the threshold raised 1 Check(s)')
    expect(summary).toHaveTextContent('Acknowledge them below')
  })

  it('shows the Checks panel on the Training Run, and it gates Continue', async () => {
    const acked = { done: false }
    const check = {
      id: 'c1',
      kind: 'fairness_gap',
      severity: 'warning' as const,
      message: REPORT.checks_raised[0].message,
      subject_type: 'training_run' as const,
      subject_id: 'j1',
      details: { gap: 'tpr' },
      acknowledged: false,
      acknowledged_at: null,
      note: null,
    }
    const user = userEvent.setup()
    mockFetch({
      ...baseHandlers,
      'GET /checks': () => ({
        checks: [{ ...check, acknowledged: acked.done }],
        unacknowledged_warnings: acked.done ? 0 : 1,
      }),
      'POST /checks/c1/acknowledge': () => {
        acked.done = true
        return { ...check, acknowledged: true, acknowledged_at: '2024-01-01T00:00:00Z' }
      },
    })
    renderWithProviders(<App />, { route: '/projects/p1/train?version=v1' })
    await user.click(await screen.findByLabelText('is_churn', { selector: '#target-is_churn' }))
    await user.click(await screen.findByRole('button', { name: 'See the plan' }))
    await screen.findByTestId('train-plan')
    await user.click(screen.getByRole('button', { name: 'Train' }))
    await screen.findByText('Training Run')
    await emitJobEvent('j1', 'completed', { status: 'completed', progress: {}, result: RUN, error: null })
    await user.click(await screen.findByRole('button', { name: 'Fairness Report' }))
    await user.click(screen.getByRole('button', { name: 'Measure this group' }))
    await screen.findByTestId('fairness-headline')

    // the same sentence appears in the gap explanation too, so scope to the panel
    const panel = (await screen.findAllByText(REPORT.checks_raised[0].message)).find((node) =>
      Boolean(node.closest('[data-slot="card"]')),
    )
    expect(panel).toBeDefined()
    // several Checks panels share this page, so the run's own is the one that
    // names the fairness Check
    const runPanel = (await screen.findAllByText(REPORT.checks_raised[0].message))[0]
      .closest('[data-slot="card"]') as HTMLElement
    const gate = within(runPanel).getByRole('button', { name: 'Continue' })
    expect(gate).toBeDisabled()
    await user.click(within(runPanel).getByRole('button', { name: 'Acknowledge' }))
    await user.click(screen.getByRole('button', { name: 'Record Acknowledgement' }))
    await waitFor(() => expect(within(runPanel).getByRole('button', { name: 'Continue' })).toBeEnabled())
  })

  it('shows no Checks panel when nothing was flagged', async () => {
    const user = userEvent.setup()
    await showReport(user, {
      'POST /train/runs/j1/fairness': () => ({
        ...REPORT,
        gaps_exceeding_threshold: [],
        checks_raised: [],
        gaps: REPORT.gaps.map((g) => ({ ...g, exceeds: false })),
      }),
    })
    await screen.findByTestId('fairness-gaps')
    expect(screen.queryByTestId('checks-raised-summary')).not.toBeInTheDocument()
  })

  it('does not open the panel until the button is pressed', async () => {
    const user = userEvent.setup()
    await showReport(user)
    // the button toggles, so a second press closes it again
    const toggle = screen.getByRole('button', { name: 'Fairness Report' })
    expect(toggle).toHaveAttribute('aria-expanded', 'true')
    await user.click(toggle)
    await waitFor(() => expect(screen.queryByTestId('fairness-panel')).not.toBeInTheDocument())
  })
})

describe('explainGap', () => {
  it('names both ends of the gap, not just its size', () => {
    const text = explainGap(REPORT.gaps[0], REPORT)
    expect(text).toContain('Accuracy difference')
    expect(text).toContain('str:a')
    expect(text).toContain('str:b')
    expect(text).toContain('over the 0.1 threshold')
  })

  it('says a gap it could not measure, with the reason', () => {
    const text = explainGap(
      { ...REPORT.gaps[0], value: null, reason: 'only one group had a value' },
      REPORT,
    )
    expect(text).toContain('could not be compared')
    expect(text).toContain('only one group had a value')
  })

  it('refuses to compare a gap the threshold cannot judge', () => {
    const text = explainGap(
      { ...REPORT.gaps[0], comparable: false, exceeds: null, unit: 'Target units' },
      REPORT,
    )
    expect(text).toContain('cannot judge a gap in Target units')
  })
})

describe('headlineGap', () => {
  it('prefers the widest gap a threshold can judge', () => {
    expect(headlineGap(REPORT)?.name).toBe('accuracy')
  })

  it('falls back to the widest of any kind', () => {
    expect(headlineGap({ ...REPORT, largest_comparable_gap: null })?.name).toBe('accuracy')
  })
})
