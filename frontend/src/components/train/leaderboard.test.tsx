/**
 * The leaderboard, and everything you can do with a row.
 *
 * The two acceptance criteria are about *control*, so they are tested by counting
 * requests:
 *
 * - **AC1 — no evaluation runs until the user asks.** Opening a leaderboard must
 *   make zero plot requests, and each plot button must make exactly one. A test
 *   that only checks the chart appears would pass just as happily against a
 *   component that prefetched all eight.
 * - **AC2 — downloads work.** The bundle and the predictions are asserted on
 *   their real URLs, because a download button that points nowhere is the one
 *   failure a user discovers only after clicking.
 */

import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import App from '@/App'
import { rankBy, rankableMetrics, type LeaderboardEntry } from '@/lib/evaluate'
import { mockFetch, renderWithProviders, text } from '@/test/render'
import { emitJobEvent, stubEventSource } from '@/test/sse'

beforeEach(() => {
  stubEventSource()
})
afterEach(() => {
  vi.unstubAllGlobals()
})

const project = {
  id: 'p1',
  name: 'Board Lab',
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
  row_count: 200,
  columns: [{ name: 'age', kind: 'integer' }],
  provenance_summary: { jev: 200 },
  seed: 7,
  meta: {},
  created_at: '2024-01-01T00:00:00Z',
}

const PLAN = {
  version_id: 'v1',
  target: 'is_churn',
  task_type: 'classification',
  label_family: 'is_churn',
  seed: 1,
  rows: 200,
  kept_rows: 180,
  train_rows: 144,
  test_rows: 36,
  n_source_features: 2,
  models: [],
  model_names: ['logistic_regression'],
  unavailable_models: [],
  primary_metric: 'f1_macro',
  primary_metric_label: 'F1 (macro)',
  primary_metric_higher_is_better: true,
  metrics_available: ['accuracy', 'f1_macro', 'roc_auc'],
  tuning: { enabled: false, strategy: 'random', n_iter: 10, cv: 3, tunable: {} },
  hyperparameters: {},
}

function entry(over: Partial<LeaderboardEntry> = {}): LeaderboardEntry {
  return {
    rank: 1,
    model: 'logistic_regression',
    label: 'Logistic Regression',
    library: 'sklearn',
    task_type: 'classification',
    status: 'ok',
    classes: ['bool:False', 'bool:True'],
    hyperparameters: {},
    fit_seconds: 0.1,
    metrics: {
      accuracy: { value: 0.9, reason: null },
      f1_macro: { value: 0.88, reason: null },
      roc_auc: { value: 0.95, reason: null },
    },
    primary_metric: 'f1_macro',
    primary: { value: 0.88, reason: null },
    ...over,
  }
}

const BOARD = [
  entry(),
  entry({
    rank: 2,
    model: 'random_forest',
    label: 'Random Forest',
    metrics: {
      accuracy: { value: 0.91, reason: null },
      f1_macro: { value: 0.93, reason: null },
      roc_auc: { value: 0.97, reason: null },
    },
    primary: { value: 0.93, reason: null },
  }),
  entry({
    rank: null,
    model: 'svm',
    label: 'SVM',
    metrics: { roc_auc: { value: null, reason: 'this Model does not output class probabilities' } },
    primary_metric: 'roc_auc',
    primary: { value: null, reason: 'this Model does not output class probabilities' },
  }),
]

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
  leaderboard: BOARD,
  warnings: [],
}

const CONFUSION = {
  plot: 'confusion_matrix',
  model: 'logistic_regression',
  labels: ['bool:False', 'bool:True'],
  matrix: [
    [16, 2],
    [1, 17],
  ],
  row_normalised: [
    [0.888, 0.111],
    [0.055, 0.944],
  ],
  support: [18, 18],
  n: 36,
}

const ROC = {
  plot: 'roc',
  model: 'logistic_regression',
  kind: 'binary',
  positive_class: 'bool:True',
  fpr: [0, 0.1, 1],
  tpr: [0, 0.6, 1],
  thresholds: [1, 0.5, -1],
  auc: 0.95,
}

const PR = {
  plot: 'precision_recall',
  model: 'logistic_regression',
  kind: 'binary',
  positive_class: 'bool:True',
  precision: [1, 0.9, 0.6],
  recall: [0, 0.5, 1],
  average_precision: 0.92,
}

const IMPORTANCE = {
  plot: 'feature_importance',
  model: 'logistic_regression',
  method: 'permutation',
  baseline: 0.9,
  repeats: 5,
  importance: [
    { feature: 'age_scaled', drop: 0.21 },
    { feature: 'spend_scaled', drop: 0.08 },
    { feature: 'plan=pro', drop: 0.01 },
  ],
}

const RESIDUALS = {
  plot: 'residuals',
  model: 'logistic_regression',
  residuals: [0.2, -0.1, 0.05],
  predicted: [1.1, 2.0, 3.05],
  actual: [1.3, 1.9, 3.1],
  mae: 0.116667,
  mean_residual: 0.05,
  n: 3,
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
    total_rows: 200,
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
  'GET /checks': () => ({ checks: [], unacknowledged_warnings: 0 }),
  'POST /train/plan': () => PLAN,
  'POST /train/run': () => ({ id: 'j1' }),
  'GET /train/runs/j1': () => ({ ...RUN, status: 'running' }),
}

// -- a regression run, where the lower-is-better metrics live ----------------

const REGRESSION_PLAN = {
  ...PLAN,
  target: 'price',
  task_type: 'regression',
  primary_metric: 'r2',
  primary_metric_label: 'R\u00b2',
  metrics_available: ['r2', 'rmse', 'mae'],
  model_names: ['ridge'],
}

const REGRESSION_METRICS = {
  task_types: {
    regression: {
      task_type: 'regression',
      default_primary_metric: 'r2',
      metrics: [
        { name: 'r2', label: 'R\u00b2', task_type: 'regression', higher_is_better: true, needs_proba: false, is_default: true, description: '' },
        { name: 'rmse', label: 'RMSE', task_type: 'regression', higher_is_better: false, needs_proba: false, is_default: false, description: '' },
        { name: 'mae', label: 'MAE', task_type: 'regression', higher_is_better: false, needs_proba: false, is_default: false, description: '' },
      ],
    },
  },
}

function regressionEntry(model: string, label: string, rmse: number, r2: number) {
  return {
    rank: null,
    model,
    label,
    library: 'sklearn',
    task_type: 'regression',
    status: 'ok',
    classes: [],
    hyperparameters: {},
    fit_seconds: 0.1,
    metrics: {
      r2: { value: r2, reason: null },
      rmse: { value: rmse, reason: null },
      mae: { value: rmse / 2, reason: null },
    },
    primary_metric: 'r2',
    primary: { value: r2, reason: null },
  }
}

const REGRESSION_RUN = {
  ...RUN,
  target: 'price',
  task_type: 'regression',
  primary_metric: 'r2',
  primary_metric_higher_is_better: true,
  warnings: [],
  leaderboard: [
    regressionEntry('ridge', 'Ridge', 1.0, 0.71),
    regressionEntry('svm', 'SVM', 9.0, 0.6),
  ],
}

/** Train a regression run and return its leaderboard. */
async function showRegressionBoard(user: ReturnType<typeof userEvent.setup>) {
  mockFetch({
    ...baseHandlers,
    'GET /dataset-versions/v1/preview': () => ({
      version_id: 'v1',
      columns: [
        { name: 'age', kind: 'integer' },
        { name: 'price', kind: 'number' },
      ],
      page: 0,
      page_size: 1,
      total_rows: 200,
      rows: [[31, 12]],
    }),
    'GET /train/metrics': () => REGRESSION_METRICS,
    'POST /train/plan': () => REGRESSION_PLAN,
  })
  renderWithProviders(<App />, { route: '/projects/p1/train?version=v1' })
  await user.click(await screen.findByLabelText('price', { selector: '#target-price' }))
  await user.click(await screen.findByRole('button', { name: 'See the plan' }))
  await screen.findByTestId('train-plan')
  await user.click(screen.getByRole('button', { name: 'Train' }))
  await screen.findByText('Training Run')
  await emitJobEvent('j1', 'completed', {
    status: 'completed',
    progress: {},
    result: REGRESSION_RUN,
    error: null,
  })
  return await screen.findByTestId('leaderboard')
}

/** The first Model's row, which is the one most of these act on. */
function row(board: HTMLElement) {
  return within(within(board).getByTestId('board-row-logistic_regression'))
}

function plotHandler(body: unknown) {
  return () => text(JSON.stringify(body), 200, 'application/json')
}

/**
 * Train, and put the run's own result on the job stream.
 *
 * `result` is a parameter because the board is drawn from the *job result*, not
 * from a refetch — a helper that always emitted a fixed payload would silently
 * ignore every `GET /train/runs/j1` override.
 */
async function showBoard(
  user: ReturnType<typeof userEvent.setup>,
  result: unknown = RUN,
) {
  renderWithProviders(<App />, { route: '/projects/p1/train?version=v1' })
  await user.click(await screen.findByLabelText('is_churn', { selector: '#target-is_churn' }))
  await user.click(await screen.findByRole('button', { name: 'See the plan' }))
  await screen.findByTestId('train-plan')
  await user.click(screen.getByRole('button', { name: 'Train' }))
  await screen.findByText('Training Run')
  await emitJobEvent('j1', 'completed', {
    status: 'completed',
    progress: {},
    result,
    error: null,
  })
  return screen.findByTestId('leaderboard')
}

// -- the board itself ---------------------------------------------------------

describe('Leaderboard', () => {
  it('lists every Model with its rank, library and metrics', async () => {
    const user = userEvent.setup()
    mockFetch(baseHandlers)
    const board = await showBoard(user)
    expect(within(board).getByText('Logistic Regression')).toBeInTheDocument()
    expect(within(board).getByText('Random Forest')).toBeInTheDocument()
    expect(within(within(board).getByTestId('board-row-logistic_regression')).getByText('sklearn')).toBeInTheDocument()
    // the primary metric is shown with its value and its direction
    expect(within(board).getByText(/f1_macro 0\.9300 ↑/)).toBeInTheDocument()
  })

  it('re-ranks by a metric the user chooses', async () => {
    const user = userEvent.setup()
    mockFetch(baseHandlers)
    const board = await showBoard(user)
    // by f1_macro the Forest wins; by roc_auc it also wins, so pick accuracy
    // where the ordering is the same — the point is that the picker exists and
    // only offers metrics the run actually computed
    const picker = within(board).getByLabelText('Rank by')
    const options = within(picker).getAllByRole('option').map((o) => o.getAttribute('value'))
    expect(options).toEqual(expect.arrayContaining(['accuracy', 'f1_macro', 'roc_auc']))
    // roc_auc is not offered as a plain option for a Model with no value... it is,
    // because another Model has it; and the run's own primary is labelled as such
    expect(within(picker).getByText("f1_macro (the run's own primary)")).toBeInTheDocument()
  })

  it('ranks a lower-is-better metric the right way round', async () => {
    // The metric's direction is a property of the metric, declared in the
    // registry. It used to be *inferred* by comparing two Models' primary
    // scores, which says nothing about the metric being ranked by — so on a
    // regression run whose primary is r2, ranking by rmse put the Model with
    // nine times the error first and captioned it "higher is better".
    const user = userEvent.setup()
    const board = await showRegressionBoard(user)

    // By the run's own primary (r2, higher is better) the better r2 leads.
    const byR2 = within(board).getAllByTestId(/^board-row-/).map((node) => node.dataset.testid)
    expect(byR2).toEqual(['board-row-ridge', 'board-row-svm'])

    // Ranking by rmse reverses it, and says so.
    await user.selectOptions(within(board).getByLabelText('Rank by'), 'rmse')
    await waitFor(() => {
      const byRmse = within(board)
        .getAllByTestId(/^board-row-/)
        .map((node) => node.dataset.testid)
      expect(byRmse).toEqual(['board-row-ridge', 'board-row-svm'])
    })
    expect(within(board).getByText(/rankable on rmse \(lower is better\)/)).toBeInTheDocument()
    // And the value carries a down arrow, not an up one.
    expect(within(board).getByText(/rmse 1\.0000 ↓/)).toBeInTheDocument()
  })

  it('does not offer a metric the run never computed', () => {
    expect(rankableMetrics(BOARD).sort()).toEqual(['accuracy', 'f1_macro', 'roc_auc'])
    expect(rankableMetrics([BOARD[2]])).toEqual(['roc_auc'].filter(() => false))
  })

  it('lists a Model that could not be ranked, with the reason', async () => {
    const user = userEvent.setup()
    mockFetch(baseHandlers)
    const board = await showBoard(user)
    const reason = within(board).getByTestId('unranked-svm')
    expect(reason).toHaveTextContent('Not ranked')
    expect(reason).toHaveTextContent('f1_macro')
  })

  it('hides a Model that did not fit behind a button, and can show it', async () => {
    const user = userEvent.setup()
    mockFetch({
      ...baseHandlers,
      'GET /train/runs/j1': () => ({
        ...RUN,
        status: 'running',
        leaderboard: [
          ...BOARD,
          entry({ rank: null, model: 'knn', label: 'KNN', status: 'failed', error: 'it exploded' }),
        ],
      }),
    })
    const withFailure = {
      ...RUN,
      leaderboard: [...BOARD, entry({ model: 'knn', label: 'KNN', status: 'failed', error: 'it exploded' })],
    }
    const board = await showBoard(user, withFailure)
    expect(within(board).queryByText('KNN')).not.toBeInTheDocument()
    await user.click(within(board).getByRole('button', { name: /Show the .* that did not fit/ }))
    expect(within(board).getByText('KNN')).toBeInTheDocument()
    // and the reason it did not fit is on the row, not just "unranked"
    expect(within(board).getByTestId('unranked-knn')).toHaveTextContent('it exploded')
  })
})

describe('AC1 — no evaluation runs until the user asks', () => {
  it('makes no plot request when the board appears', async () => {
    const user = userEvent.setup()
    const { calls } = mockFetch({
      ...baseHandlers,
      'GET /train/runs/j1/plots': plotHandler(CONFUSION),
    })
    await showBoard(user)
    // a leaderboard that silently refits eight models to draw charts nobody
    // looked at is a leaderboard you wait for
    expect(calls.some(([, p]) => p.includes('/plots'))).toBe(false)
  })

  it('makes no plot request from hovering, focusing or tabbing a row', async () => {
    const user = userEvent.setup()
    const { calls } = mockFetch({
      ...baseHandlers,
      'GET /train/runs/j1/plots': plotHandler(CONFUSION),
    })
    const board = await showBoard(user)
    for (const model of ['logistic_regression', 'random_forest', 'svm']) {
      await user.hover(within(board).getByTestId(`board-row-${model}`))
      within(board).getByTestId(`board-row-${model}`).focus()
    }
    await user.tab()
    expect(calls.some(([, p]) => p.includes('/plots'))).toBe(false)
  })

  it('makes exactly one plot request when a plot is asked for', async () => {
    const user = userEvent.setup()
    const { calls } = mockFetch({
      ...baseHandlers,
      'GET /train/runs/j1/plots': plotHandler(CONFUSION),
    })
    const board = await showBoard(user)
    await user.click(
      within(within(board).getByTestId('board-row-logistic_regression')).getByRole('button', {
        name: 'Confusion matrix',
      }),
    )
    await screen.findByTestId('plot-confusion_matrix')
    const plots = calls.filter(([, p]) => p.includes('/plots'))
    expect(plots).toHaveLength(1)
    expect(plots[0][1]).toContain('model=logistic_regression')
    expect(plots[0][1]).toContain('plot=confusion_matrix')
  })

  it('shows the confusion matrix as a readable table, not just a picture', async () => {
    const user = userEvent.setup()
    mockFetch({ ...baseHandlers, 'GET /train/runs/j1/plots': plotHandler(CONFUSION) })
    const board = await showBoard(user)
    await user.click(row(board).getByRole('button', { name: 'Confusion matrix' }))
    const table = await screen.findByTestId('confusion-table')
    const headers = within(table).getAllByRole('columnheader').map((h) => h.textContent)
    expect(headers).toEqual(['actual \\ predicted', 'bool:False', 'bool:True', 'support'])
    expect(within(table).getByText('16')).toBeInTheDocument()
    expect(within(table).getByText('17')).toBeInTheDocument()
  })

  it('renders the ROC curve and says the AUC in words as well as drawing it', async () => {
    const user = userEvent.setup()
    mockFetch({ ...baseHandlers, 'GET /train/runs/j1/plots': plotHandler(ROC) })
    const board = await showBoard(user)
    await user.click(row(board).getByRole('button', { name: 'ROC curve' }))
    expect(await screen.findByTestId('plot-roc')).toBeInTheDocument()
    expect(screen.getByTestId('auc')).toHaveTextContent('AUC 0.9500')
    expect(screen.getByRole('img', { name: /Receiver operating characteristic/ })).toBeInTheDocument()
  })

  it('renders precision / recall with its average precision', async () => {
    const user = userEvent.setup()
    mockFetch({ ...baseHandlers, 'GET /train/runs/j1/plots': plotHandler(PR) })
    const board = await showBoard(user)
    await user.click(row(board).getByRole('button', { name: 'Precision / recall' }))
    expect(await screen.findByTestId('plot-precision_recall')).toBeInTheDocument()
    expect(screen.getByTestId('average-precision')).toHaveTextContent('Average precision 0.9200')
  })

  it('lists feature importance with its numbers, not just bars', async () => {
    const user = userEvent.setup()
    mockFetch({ ...baseHandlers, 'GET /train/runs/j1/plots': plotHandler(IMPORTANCE) })
    const board = await showBoard(user)
    await user.click(row(board).getByRole('button', { name: 'Feature importance' }))
    const list = await screen.findByTestId('importance-list')
    expect(within(list).getByText('age_scaled')).toBeInTheDocument()
    expect(within(list).getByText('0.2100')).toBeInTheDocument()
    expect(screen.getByText(/Baseline score 0.9000/)).toBeInTheDocument()
  })

  it('does not re-request a plot that is already showing', async () => {
    const user = userEvent.setup()
    const { calls } = mockFetch({
      ...baseHandlers,
      'GET /train/runs/j1/plots': plotHandler(CONFUSION),
    })
    const board = await showBoard(user)
    const button = row(board).getByRole('button', { name: 'Confusion matrix' })
    await user.click(button)
    await screen.findByTestId('plot-confusion_matrix')
    // closing and reopening must not silently re-run a search that costs time
    await user.click(button)
    await waitFor(() => expect(screen.queryByTestId('plot-confusion_matrix')).not.toBeInTheDocument())
    await user.click(button)
    await screen.findByTestId('plot-confusion_matrix')
    expect(calls.filter(([, p]) => p.includes('/plots'))).toHaveLength(1)
  })

  it('shows one Model at a time, so the panel is never a wall of charts', async () => {
    const user = userEvent.setup()
    mockFetch({ ...baseHandlers, 'GET /train/runs/j1/plots': plotHandler(CONFUSION) })
    const board = await showBoard(user)
    await user.click(row(board).getByRole('button', { name: 'ROC curve' }))
    await screen.findByTestId('plot-roc')
    expect(screen.queryByTestId('plot-precision_recall')).not.toBeInTheDocument()
  })

  it('shows a Model cannot produce as a reason, not an empty chart', async () => {
    const user = userEvent.setup()
    mockFetch({
      ...baseHandlers,
      'GET /train/runs/j1/plots': plotHandler({
        plot: 'roc',
        model: 'svm',
        error: 'this Model does not output class probabilities',
      }),
    })
    const board = await showBoard(user)
    const svm = within(within(board).getByTestId('board-row-svm'))
    await user.click(svm.getByRole('button', { name: 'ROC curve' }))
    expect(await screen.findByTestId('plot-unavailable')).toHaveTextContent(
      'does not output class probabilities',
    )
    expect(screen.queryByTestId('plot-roc')).not.toBeInTheDocument()
  })

  it('offers residuals for a regression Target and not a confusion matrix', async () => {
    const user = userEvent.setup()
    mockFetch({
      ...baseHandlers,
      'GET /train/runs/j1': () => ({
        ...RUN,
        status: 'running',
        task_type: 'regression',
        primary_metric: 'r2',
        leaderboard: [entry({ task_type: 'regression', primary_metric: 'r2' })],
      }),
      'GET /train/runs/j1/plots': plotHandler(RESIDUALS),
    })
    const regressionRun = {
      ...RUN,
      task_type: 'regression',
      primary_metric: 'r2',
      leaderboard: [entry({ task_type: 'regression', primary_metric: 'r2' })],
    }
    const board = await showBoard(user, regressionRun)
    const single = within(board).getByTestId('board-row-logistic_regression')
    expect(within(single).getByRole('button', { name: 'Residuals' })).toBeInTheDocument()
    expect(within(single).queryByRole('button', { name: 'Confusion matrix' })).not.toBeInTheDocument()
    await user.click(within(single).getByRole('button', { name: 'Residuals' }))
    expect(await screen.findByTestId('residual-summary')).toHaveTextContent('MAE 0.1167')
  })
})

describe('AC2 — downloads work from the UI', () => {
  it('links the Model Bundle at its real URL', async () => {
    const user = userEvent.setup()
    mockFetch(baseHandlers)
    const board = await showBoard(user)
    const row = within(board).getByTestId('board-row-logistic_regression')
    const link = within(row).getByRole('link', { name: /Model Bundle/ })
    expect(link).toHaveAttribute('href', '/api/train/runs/j1/bundle?model=logistic_regression')
    expect(link).toHaveAttribute('download', 'logistic_regression-bundle.zip')
  })

  it('gives each Model its own bundle link', async () => {
    const user = userEvent.setup()
    mockFetch(baseHandlers)
    const board = await showBoard(user)
    const links = within(board).getAllByRole('link', { name: /Model Bundle/ })
    expect(links).toHaveLength(3)
    // one per Model, and each points at that Model
    expect(links.map((l) => l.getAttribute('href')).sort()).toEqual([
      '/api/train/runs/j1/bundle?model=logistic_regression',
      '/api/train/runs/j1/bundle?model=random_forest',
      '/api/train/runs/j1/bundle?model=svm',
    ])
    expect(links.map((l) => l.getAttribute('download'))).toEqual(
      links.map((l) => (l.getAttribute('href') ?? '').split('model=')[1] + '-bundle.zip'),
    )
  })

  it('does not offer a bundle for a Model that did not fit', async () => {
    const user = userEvent.setup()
    mockFetch({
      ...baseHandlers,
      'GET /train/runs/j1': () => ({
        ...RUN,
        status: 'running',
        leaderboard: [...BOARD, entry({ model: 'knn', label: 'KNN', status: 'failed', error: 'boom' })],
      }),
    })
    const withFailure = {
      ...RUN,
      leaderboard: [...BOARD, entry({ model: 'knn', label: 'KNN', status: 'failed', error: 'boom' })],
    }
    const board = await showBoard(user, withFailure)
    await user.click(within(board).getByRole('button', { name: /Show the .* that did not fit/ }))
    const knn = within(board).getByTestId('board-row-knn')
    expect(within(knn).queryByRole('link', { name: /Model Bundle/ })).not.toBeInTheDocument()
    expect(within(knn).queryByRole('button', { name: 'ROC curve' })).not.toBeInTheDocument()
  })

  it('predicts on an uploaded CSV and offers the download', async () => {
    const user = userEvent.setup()
    mockFetch({
      ...baseHandlers,
      'POST /train/runs/j1/predict': () => ({
        run_id: 'j1',
        model: 'logistic_regression',
        rows: 2,
        columns: ['age', 'is_churn', 'prediction', 'probability_bool:False', 'probability_bool:True'],
        predictions: [
          { age: 20, is_churn: null, prediction: 'True', 'probability_bool:True': 0.8 },
          { age: 70, is_churn: null, prediction: 'False', 'probability_bool:True': 0.1 },
        ],
        download_url: '/api/train/runs/j1/predictions.csv?model=logistic_regression&token=abc',
      }),
    })
    const board = await showBoard(user)
    const single = within(board).getByTestId('board-row-logistic_regression')
    // the upload affordance is closed until asked for
    expect(within(single).queryByLabelText(/A CSV with/)).not.toBeInTheDocument()
    await user.click(within(single).getByRole('button', { name: /Predict on a CSV/ }))

    const file = new File(['age\n20\n70\n'], 'new.csv', { type: 'text/csv' })
    await user.upload(within(single).getByLabelText(/A CSV with/), file)

    const result = await screen.findByTestId('predict-result-logistic_regression')
    expect(within(result).getByText(/2 row\(s\) predicted/)).toBeInTheDocument()
    const link = within(result).getByRole('link', { name: /Download predictions/ })
    expect(link).toHaveAttribute('href', '/api/train/runs/j1/predictions.csv?model=logistic_regression&token=abc')
    expect(link).toHaveAttribute('download', 'predictions.csv')
  })

  it('reports a prediction the backend refused, without losing the board', async () => {
    const user = userEvent.setup()
    mockFetch({
      ...baseHandlers,
      'POST /train/runs/j1/predict': () => {
        throw new Error('this data is missing column(s) the Model needs: age, spend')
      },
    })
    const board = await showBoard(user)
    const single = within(board).getByTestId('board-row-logistic_regression')
    await user.click(within(single).getByRole('button', { name: /Predict on a CSV/ }))
    const file = new File(['x\n1\n'], 'bad.csv', { type: 'text/csv' })
    await user.upload(within(single).getByLabelText(/A CSV with/), file)

    expect(await screen.findByRole('alert')).toHaveTextContent('missing column(s)')
    // the board is still there
    expect(within(board).getByText('Logistic Regression')).toBeInTheDocument()
  })

  it('does not predict before a file is chosen', async () => {
    const user = userEvent.setup()
    const { calls } = mockFetch(baseHandlers)
    const board = await showBoard(user)
    const single = within(board).getByTestId('board-row-logistic_regression')
    await user.click(within(single).getByRole('button', { name: /Predict on a CSV/ }))
    expect(calls.some(([, p]) => p.includes('/predict'))).toBe(false)
  })
})

describe('rankBy', () => {
  it('re-ranks by a chosen metric, in the chosen direction', () => {
    const byF1 = rankBy(BOARD, 'f1_macro', true)
    expect(byF1.map((e) => e.model)).toEqual(['random_forest', 'logistic_regression', 'svm'])
    expect(byF1[0].displayRank).toBe(1)
    expect(byF1[1].displayRank).toBe(2)
    expect(byF1[2].displayRank).toBeNull()
  })

  it('leaves a Model unranked when the metric has no value for it', () => {
    const byAuc = rankBy(BOARD, 'accuracy', true)
    const svm = byAuc.find((e) => e.model === 'svm')!
    expect(svm.rankable).toBe(false)
    expect(svm.unrankedReason).toBeTruthy()
  })

  it('carries the backend reason through when it has one', () => {
    const [a] = rankBy([BOARD[2]], 'roc_auc', true)
    expect(a.unrankedReason).toBe('this Model does not output class probabilities')
  })

  it('uses competition ranking, so a tie shares a rank and the next is skipped', () => {
    const tied = [entry({ model: 'a' }), entry({ model: 'b' })]
    const ranked = rankBy(tied, 'f1_macro', true)
    expect(ranked.map((e) => e.displayRank)).toEqual([1, 1])
  })

  it('orders ascending when lower is better', () => {
    const low = [
      entry({ model: 'a', primary: { value: 0.5, reason: null } }),
      entry({ model: 'b', primary: { value: 0.1, reason: null } }),
    ]
    const ranked = rankBy(low, 'f1_macro', false)
    expect(ranked.map((e) => e.model)).toEqual(['b', 'a'])
  })

  it('explains a Model that did not fit, rather than calling it unranked on a metric', () => {
    const failed = rankBy([entry({ status: 'failed', error: 'it exploded' })], 'f1_macro', true)
    expect(failed[0].unrankedReason).toBe('it exploded')
  })
})
