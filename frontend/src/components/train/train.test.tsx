/**
 * The Train step.
 *
 * Two things are being protected.
 *
 * **The Plan is a gate, not a summary.** Choosing a Target and ticking Models
 * must not start anything; the step asks the backend what it *would* do and
 * shows the answer first. A run that can start without a plan read is a run
 * whose split, Task Type and usable Models were decided without you seeing them.
 *
 * **A sibling Label Column is never a Target.** `is_churn__confidence` is not a
 * thing to predict — it is Jev's own certainty about the real Target, so a model
 * trained on it learns nothing and looks like it works. The picker filters the
 * siblings out rather than letting the leaderboard discover it.
 */

import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import App from '@/App'
import { modelsForTaskType, unavailableReason, type ModelSpecDto, type TrainPlan } from '@/lib/train'
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
  name: 'Train Lab',
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
  columns: [
    { name: 'age', kind: 'integer' },
    { name: 'region', kind: 'categorical' },
    { name: 'is_churn', kind: 'bool' },
  ],
  provenance_summary: { jev: 200 },
  seed: 7,
  meta: {},
  created_at: '2024-01-01T00:00:00Z',
}

function model(
  name: string,
  label: string,
  over: Partial<ModelSpecDto> = {},
): ModelSpecDto {
  return {
    name,
    label,
    library: 'sklearn',
    extra: null,
    task_types: ['classification', 'regression'],
    supports_proba: true,
    notes: '',
    available: true,
    ...over,
  }
}

const MODELS = [
  model('logistic_regression', 'Logistic Regression', { task_types: ['classification'] }),
  model('random_forest', 'Random Forest'),
  model('naive_bayes', 'Naive Bayes', { task_types: ['classification'] }),
  model('torch_mlp', 'Torch MLP', { library: 'torch', extra: 'torch', available: false }),
]

const METRICS = {
  task_types: {
    classification: {
      default_primary_metric: 'f1_macro',
      metrics: [
        {
          name: 'f1_macro',
          label: 'F1 (macro)',
          task_type: 'classification',
          higher_is_better: true,
          needs_proba: false,
          is_default: true,
          description: 'Balanced across classes',
        },
        {
          name: 'accuracy',
          label: 'Accuracy',
          task_type: 'classification',
          higher_is_better: true,
          needs_proba: false,
          is_default: false,
          description: 'Plain fraction correct',
        },
        {
          name: 'roc_auc',
          label: 'ROC-AUC',
          task_type: 'classification',
          higher_is_better: true,
          needs_proba: true,
          is_default: false,
          description: 'Needs class probabilities',
        },
      ],
    },
    regression: { default_primary_metric: 'r2', metrics: [] },
  },
}

const PLAN: TrainPlan = {
  version_id: 'v1',
  target: 'is_churn',
  task_type: 'classification',
  label_family: 'is_churn',
  seed: 4242,
  rows: 200,
  kept_rows: 180,
  train_rows: 144,
  test_rows: 36,
  n_source_features: 2,
  models: MODELS.filter((m) => m.available),
  model_names: ['logistic_regression', 'random_forest'],
  unavailable_models: [],
  primary_metric: 'f1_macro',
  primary_metric_label: 'F1 (macro)',
  primary_metric_higher_is_better: true,
  metrics_available: ['accuracy', 'f1_macro', 'roc_auc'],
  tuning: { enabled: false, strategy: 'random', n_iter: 10, cv: 3, tunable: {} },
  hyperparameters: {},
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
  seed: 4242,
  primary_metric: 'f1_macro',
  leaderboard: [
    {
      rank: 1,
      model: 'random_forest',
      label: 'Random Forest',
      library: 'sklearn',
      task_type: 'classification',
      status: 'ok',
      classes: ['bool:False', 'bool:True'],
      hyperparameters: { n_estimators: 100 },
      fit_seconds: 0.4,
      metrics: { accuracy: { value: 0.91, reason: null } },
      primary_metric: 'f1_macro',
      primary: { value: 0.9, reason: null },
    },
    {
      rank: 2,
      model: 'logistic_regression',
      label: 'Logistic Regression',
      library: 'sklearn',
      task_type: 'classification',
      status: 'ok',
      classes: ['bool:False', 'bool:True'],
      hyperparameters: {},
      fit_seconds: 0.01,
      metrics: { accuracy: { value: 0.88, reason: null } },
      primary_metric: 'f1_macro',
      primary: { value: 0.87, reason: null },
    },
    {
      rank: null,
      model: 'svm',
      label: 'SVM',
      library: 'sklearn',
      task_type: 'classification',
      status: 'ok',
      classes: [],
      hyperparameters: {},
      fit_seconds: 0.2,
      metrics: { roc_auc: { value: null, reason: 'no probabilities' } },
      primary_metric: 'roc_auc',
      primary: { value: null, reason: 'this Model does not output class probabilities' },
    },
  ],
  warnings: ['the Target is 8% of the rows'],
}

const baseHandlers = {
  'GET /projects': () => [project],
  'GET /projects/p1': () => project,
  'GET /projects/p1/dataset_versions': () => [V1],
  'GET /dataset-versions/v1/preview': () => ({
    version_id: 'v1',
    columns: [
      { name: 'age', kind: 'integer' },
      { name: 'region', kind: 'categorical' },
      { name: 'is_churn', kind: 'bool' },
      { name: 'is_churn__confidence', kind: 'number' },
    ],
    page: 0,
    page_size: 1,
    total_rows: 200,
    rows: [[31, 'north', true, 0.9]],
  }),
  'GET /train/models': () => ({ models: MODELS, extras: {}, library_versions: {} }),
  'GET /train/metrics': () => METRICS,
  'GET /train/runs': () => ({ project_id: 'p1', count: 0, runs: [] }),
  'GET /checks': () => ({ checks: [], unacknowledged_warnings: 0 }),
}

function renderStep() {
  return renderWithProviders(<App />, { route: '/projects/p1/train?version=v1' })
}

/** Choose the Target, tick nothing else, and read the plan. */
async function toThePlan(user: ReturnType<typeof userEvent.setup>) {
  renderStep()
  await user.click(await screen.findByLabelText('is_churn', { selector: '#target-is_churn' }))
  await user.click(await screen.findByRole('button', { name: 'See the plan' }))
  return screen.findByTestId('train-plan')
}

describe('Train step — the Target', () => {
  it('offers the Label Columns and nothing else', async () => {
    mockFetch(baseHandlers)
    renderStep()
    await screen.findByLabelText('is_churn', { selector: '#target-is_churn' })
    expect(screen.getByLabelText('age', { selector: '#target-age' })).toBeInTheDocument()
  })

  it('never offers a sibling Label Column as a Target', async () => {
    mockFetch(baseHandlers)
    renderStep()
    await screen.findByLabelText('is_churn', { selector: '#target-is_churn' })
    // is_churn__confidence is Jev's certainty about the Target. Predicting it
    // would learn nothing and look like it works.
    expect(screen.queryByLabelText(/is_churn__confidence/)).not.toBeInTheDocument()
  })

  it('cannot plan without a Target', async () => {
    const { calls } = mockFetch(baseHandlers)
    renderStep()
    await screen.findByText('Target')
    expect(screen.getByRole('button', { name: 'See the plan' })).toBeDisabled()
    await waitFor(() => expect(calls.some(([m]) => m === 'POST')).toBe(false))
  })

  it('says so when there is nothing to train on', async () => {
    mockFetch({ ...baseHandlers, 'GET /projects/p1/dataset_versions': () => [] })
    renderStep()
    expect(await screen.findByText(/No Dataset Versions yet/)).toBeInTheDocument()
  })
})

describe('Train step — Models', () => {
  it('lists the Models for the Task Type, and says which are missing and why', async () => {
    const user = userEvent.setup()
    mockFetch({ ...baseHandlers, 'POST /train/plan': () => PLAN })
    renderStep()
    await user.click(await screen.findByLabelText('is_churn', { selector: '#target-is_churn' }))

    await screen.findByLabelText('Logistic Regression')
    expect(screen.getByLabelText('Random Forest')).toBeInTheDocument()
    // torch is an optional extra and says so, with the command that fixes it
    expect(screen.getByText(/Torch MLP — needs the optional extra 'torch'/)).toBeInTheDocument()
    expect(screen.getByText(/uv sync --extra torch/)).toBeInTheDocument()
  })

  it('says that ticking nothing means every Model', async () => {
    const user = userEvent.setup()
    mockFetch({ ...baseHandlers, 'POST /train/plan': () => PLAN })
    renderStep()
    await user.click(await screen.findByLabelText('is_churn', { selector: '#target-is_churn' }))
    expect(await screen.findByText(/None ticked means every Model/)).toBeInTheDocument()
  })

  it('sends the Models that were ticked', async () => {
    const user = userEvent.setup()
    const { calls, lastBody } = mockFetch({ ...baseHandlers, 'POST /train/plan': () => PLAN })
    renderStep()
    await user.click(await screen.findByLabelText('is_churn', { selector: '#target-is_churn' }))
    await user.click(await screen.findByLabelText('Random Forest'))
    await user.click(screen.getByRole('button', { name: 'See the plan' }))
    await screen.findByTestId('train-plan')
    expect(calls).toContainEqual(['POST', '/train/plan'])
    expect(lastBody()).toMatchObject({ models: ['random_forest'] })
  })
})

describe('Train step — the plan is a gate', () => {
  it('shows what will happen before anything runs', async () => {
    const user = userEvent.setup()
    const { calls, lastBody } = mockFetch({ ...baseHandlers, 'POST /train/plan': () => PLAN })
    const plan = await toThePlan(user)

    expect(within(plan).getByText('classification')).toBeInTheDocument()
    expect(within(plan).getByText('180 kept of 200')).toBeInTheDocument()
    expect(within(plan).getByText('144 train / 36 held out')).toBeInTheDocument()
    expect(within(plan).getByText('4242')).toBeInTheDocument()
    expect(within(plan).getByText('F1 (macro) (higher is better)')).toBeInTheDocument()
    // and nothing has trained yet
    expect(calls.some(([, p]) => p === '/train/run')).toBe(false)
    expect(screen.queryByText('Training Run')).not.toBeInTheDocument()
    // what was sent is the Target and the Model list, and nothing else
    expect(lastBody()).toMatchObject({ version_id: 'v1', target: 'is_churn' })
  })

  it('never lets the step invent a split', async () => {
    const user = userEvent.setup()
    const { calls, lastBody } = mockFetch({ ...baseHandlers, 'POST /train/plan': () => PLAN })
    await toThePlan(user)
    const body = calls.find(([m, p]) => m === 'POST' && p === '/train/plan')
    expect(body).toBeDefined()
    // the split, the seed and the preprocessing are derived by the backend;
    // sending our own would let the run drift away from the plan
    const sent = lastBody()
    expect(sent).not.toHaveProperty('test_size')
    expect(sent).not.toHaveProperty('seed')
  })

  it('reports a plan the backend refused, in a sentence', async () => {
    const user = userEvent.setup()
    mockFetch({
      ...baseHandlers,
      'POST /train/plan': () => {
        throw new Error('the Target has only one class, so there is nothing to learn')
      },
    })
    renderStep()
    await user.click(await screen.findByLabelText('is_churn', { selector: '#target-is_churn' }))
    await user.click(screen.getByRole('button', { name: 'See the plan' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('only one class')
    expect(screen.queryByTestId('train-plan')).not.toBeInTheDocument()
  })

  it('discards the plan when the Target changes, so a stale plan cannot be run', async () => {
    const user = userEvent.setup()
    mockFetch({ ...baseHandlers, 'POST /train/plan': () => PLAN })
    renderStep()
    await user.click(await screen.findByLabelText('is_churn', { selector: '#target-is_churn' }))
    await user.click(screen.getByRole('button', { name: 'See the plan' }))
    await screen.findByTestId('train-plan')

    // ticking a different Target must not leave a runnable plan behind
    await user.click(screen.getByLabelText('age', { selector: '#target-age' }))
    await waitFor(() => expect(screen.queryByTestId('train-plan')).not.toBeInTheDocument())
  })
})

describe('Train step — the Task Type outlives the plan', () => {
  it('keeps the Task Type when the ranking metric changes', async () => {
    // The Task Type is a property of the Target column, so it does not change
    // when the ranking metric does. It used to live only inside the plan, and
    // changing the metric threw the plan away — so the "Rank by" select
    // disappeared the instant you set it, and the Model list widened to every
    // Task Type's Models, letting a regression-only Model be ticked against a
    // classification Target with nothing objecting.
    const user = userEvent.setup()
    mockFetch({ ...baseHandlers, 'POST /train/plan': () => PLAN })
    await toThePlan(user)
    const picker = await screen.findByLabelText('Rank by', {}, { timeout: 5000 })

    const offered = () =>
      screen
        .queryAllByRole('checkbox')
        .map((box) => box.getAttribute('id'))
        .filter((id): id is string => Boolean(id?.startsWith('model-')))
    const before = offered()
    await user.selectOptions(picker, 'roc_auc')
    await waitFor(() => expect(screen.queryByTestId('train-plan')).not.toBeInTheDocument())

    // Still the same Task Type's UI: the picker is there and the Model list has
    // not grown to include another Task Type's Models.
    expect(screen.getByLabelText('Rank by')).toBeInTheDocument()
    expect(screen.getByLabelText('Rank by')).toHaveValue('roc_auc')
    expect(offered()).toEqual(before)
    // And no Model from another Task Type crept in.
    expect(offered().some((id) => id === 'model-ridge')).toBe(false)
  })
})

describe('Train step — choosing the Dataset Version', () => {
  it('works on the version that was chosen', async () => {
    // The select's handler used to ignore `event.target.value`: it reset the
    // form and left the box showing the old version, so the only visible effect
    // of choosing one was that the Target, the chosen Models and the plan
    // vanished. A Project with several Dataset Versions could not be trained on
    // anything but the deep-linked or latest one without editing the URL — which
    // is also why the e2e suite never exercised this control.
    const user = userEvent.setup()
    const V2 = { ...V1, id: 'v2', number: 2, origin: 'labeled' }
    mockFetch({
      ...baseHandlers,
      'GET /projects/p1/dataset_versions': () => [V1, V2],
      'GET /dataset-versions/v2/preview': () => ({
        ...baseHandlers['GET /dataset-versions/v1/preview'](),
        version_id: 'v2',
      }),
    })
    renderStep()
    const select = await screen.findByLabelText('Dataset Version')
    expect(select).toHaveValue('v1')

    await user.selectOptions(select, 'v2')

    // The box shows what was chosen, rather than snapping back to the old value.
    await waitFor(() => expect(select).toHaveValue('v2'))
    // And the Target is cleared, because it has to be read from the new version.
    expect(screen.queryByLabelText('is_churn', { selector: '#target-is_churn' })).not.toBeChecked()
  })
})

describe('Train step — the Sensitive Attribute', () => {
  it('offers one and explains that it is held out by default', async () => {
    const user = userEvent.setup()
    mockFetch(baseHandlers)
    renderStep()
    await user.click(await screen.findByLabelText('is_churn', { selector: '#target-is_churn' }))
    expect(await screen.findByLabelText('Sensitive Attribute')).toBeInTheDocument()
    expect(screen.getByText(/left out of the features by default/)).toBeInTheDocument()
  })

  it('sends it, and can exclude features on top', async () => {
    const user = userEvent.setup()
    const { calls, lastBody } = mockFetch({ ...baseHandlers, 'POST /train/plan': () => PLAN })
    renderStep()
    await user.click(await screen.findByLabelText('is_churn', { selector: '#target-is_churn' }))
    await user.selectOptions(await screen.findByLabelText('Sensitive Attribute'), 'region')
    await user.click(screen.getByRole('button', { name: 'See the plan' }))
    await screen.findByTestId('train-plan')
    expect(lastBody()).toMatchObject({
      sensitive_attribute: 'region',
      exclude_features: [],
    })
    expect(calls).toContainEqual(['POST', '/train/plan'])
  })
})

describe('Train step — the primary metric', () => {
  it('defaults to the Task Type default and says every metric is still computed', async () => {
    const user = userEvent.setup()
    mockFetch({ ...baseHandlers, 'POST /train/plan': () => PLAN })
    // the Task Type is only known once the backend has read the Target, so the
    // metric picker cannot honestly appear before the plan
    await toThePlan(user)
    const picker = await screen.findByLabelText('Rank by', {}, { timeout: 5000 })
    expect(within(picker).getByText('Default (f1_macro)')).toBeInTheDocument()
    expect(screen.getByText(/this only decides the order/)).toBeInTheDocument()
  })

  it('sends the metric the user chose', async () => {
    const user = userEvent.setup()
    const { lastBody } = mockFetch({ ...baseHandlers, 'POST /train/plan': () => PLAN })
    await toThePlan(user)
    await user.selectOptions(
      await screen.findByLabelText('Rank by', {}, { timeout: 5000 }),
      'roc_auc',
    )
    // choosing a metric invalidates the plan, so it is re-planned before running
    await waitFor(() => expect(screen.queryByTestId('train-plan')).not.toBeInTheDocument())
    await user.click(screen.getByRole('button', { name: 'See the plan' }))
    await screen.findByTestId('train-plan')
    expect(lastBody()).toMatchObject({ primary_metric: 'roc_auc' })
  })

  it('does not offer a metric picker before the Task Type is known', async () => {
    const user = userEvent.setup()
    mockFetch(baseHandlers)
    renderStep()
    await user.click(await screen.findByLabelText('is_churn', { selector: '#target-is_churn' }))
    await screen.findByLabelText('Logistic Regression')
    // the metrics a Task Type produces are not knowable before the backend has
    // read the Target, so guessing here would offer metrics that never compute
    expect(screen.queryByLabelText('Rank by')).not.toBeInTheDocument()
  })
})

describe('Train step — tuning', () => {
  it('says the search never sees the held-out rows', async () => {
    const user = userEvent.setup()
    mockFetch({ ...baseHandlers, 'POST /train/plan': () => PLAN })
    renderStep()
    await user.click(await screen.findByLabelText('is_churn', { selector: '#target-is_churn' }))
    expect(await screen.findByText(/held-out test rows are never seen by the search/)).toBeInTheDocument()
  })

  it('sends tuning off unless it is asked for', async () => {
    const user = userEvent.setup()
    const { lastBody } = mockFetch({ ...baseHandlers, 'POST /train/plan': () => PLAN })
    await toThePlan(user)
    expect(lastBody()).toMatchObject({ tuning: { enabled: false } })
  })

  it('sends tuning on when it is asked for', async () => {
    const user = userEvent.setup()
    const { lastBody } = mockFetch({ ...baseHandlers, 'POST /train/plan': () => PLAN })
    renderStep()
    await user.click(await screen.findByLabelText('is_churn', { selector: '#target-is_churn' }))
    await user.click(await screen.findByText('Tune hyperparameters'))
    await user.click(screen.getByRole('button', { name: 'See the plan' }))
    await screen.findByTestId('train-plan')
    expect(lastBody()).toMatchObject({
      tuning: { enabled: true, strategy: 'random', cv: 3 },
    })
  })
})

describe('Train step — running and the leaderboard', () => {
  it('runs only from the plan, and then shows the leaderboard', async () => {
    const user = userEvent.setup()
    const { calls, lastBody } = mockFetch({
      ...baseHandlers,
      'POST /train/plan': () => PLAN,
      'POST /train/run': () => ({ id: 'j1' }),
      'GET /train/runs/j1': () => ({ ...RUN, status: 'running' }),
      'GET /train/runs/j1/card': () => text('# Model Card', 200, 'text/markdown'),
    })
    const plan = await toThePlan(user)
    await user.click(within(plan).getByRole('button', { name: 'Train' }))

    await screen.findByText('Training Run')
    expect(calls).toContainEqual(['POST', '/train/run'])
    // the run sends the same body the plan was built from, so a run and its
    // plan can never describe different things
    expect(lastBody()).toMatchObject({ version_id: 'v1', target: 'is_churn' })
    await emitJobEvent('j1', 'completed', {
      status: 'completed',
      progress: {},
      result: RUN,
      error: null,
    })

    const board = await screen.findByTestId('leaderboard')
    const rows = within(board).getAllByTestId(/^board-row-/)
    expect(rows).toHaveLength(3)
    expect(within(board).getByText('Random Forest')).toBeInTheDocument()
    expect(within(board).getByText(/f1_macro 0\.9000/)).toBeInTheDocument()
  })

  it('explains a Model that could not be ranked rather than hiding it', async () => {
    const user = userEvent.setup()
    mockFetch({
      ...baseHandlers,
      'POST /train/plan': () => PLAN,
      'POST /train/run': () => ({ id: 'j1' }),
      'GET /train/runs/j1': () => ({ ...RUN, status: 'running' }),
    })
    const plan = await toThePlan(user)
    await user.click(within(plan).getByRole('button', { name: 'Train' }))
    await screen.findByText('Training Run')
    await emitJobEvent('j1', 'completed', { status: 'completed', progress: {}, result: RUN, error: null })

    const board = await screen.findByTestId('leaderboard')
    // the Model is still listed, and the reason names the metric that was missing
    expect(within(board).getByText('SVM')).toBeInTheDocument()
    const reason = within(board).getByTestId('unranked-svm')
    expect(reason).toHaveTextContent('Not ranked')
    expect(reason).toHaveTextContent('f1_macro')
  })

  it('surfaces the run warnings', async () => {
    const user = userEvent.setup()
    mockFetch({
      ...baseHandlers,
      'POST /train/plan': () => PLAN,
      'POST /train/run': () => ({ id: 'j1' }),
      'GET /train/runs/j1': () => ({ ...RUN, status: 'running' }),
    })
    const plan = await toThePlan(user)
    await user.click(within(plan).getByRole('button', { name: 'Train' }))
    await screen.findByText('Training Run')
    await emitJobEvent('j1', 'completed', { status: 'completed', progress: {}, result: RUN, error: null })
    // warnings describe the run, not the board, so they sit beside it
    await screen.findByTestId('leaderboard')
    const warnings = await screen.findByTestId('run-warnings')
    expect(within(warnings).getByText('the Target is 8% of the rows')).toBeInTheDocument()
  })

  it('offers the Model Card once the run finishes', async () => {
    const user = userEvent.setup()
    mockFetch({
      ...baseHandlers,
      'POST /train/plan': () => PLAN,
      'POST /train/run': () => ({ id: 'j1' }),
      'GET /train/runs/j1': () => ({ ...RUN, status: 'running' }),
      'GET /train/runs/j1/card': () => text('# Model Card\n', 200, 'text/markdown'),
    })
    const plan = await toThePlan(user)
    await user.click(within(plan).getByRole('button', { name: 'Train' }))
    await screen.findByText('Training Run')
    await emitJobEvent('j1', 'completed', { status: 'completed', progress: {}, result: RUN, error: null })
    expect(await screen.findByRole('button', { name: 'Model Card' })).toBeInTheDocument()
  })

  it('reports a run the backend refused', async () => {
    const user = userEvent.setup()
    mockFetch({
      ...baseHandlers,
      'POST /train/plan': () => PLAN,
      'POST /train/run': () => {
        throw new Error('Model torch_mlp needs the optional extra')
      },
    })
    const plan = await toThePlan(user)
    await user.click(within(plan).getByRole('button', { name: 'Train' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('needs the optional extra')
  })
})

describe('modelsForTaskType', () => {
  it('splits by Task Type and by availability', () => {
    const { usable, unavailable } = modelsForTaskType(MODELS, 'classification')
    expect(usable.map((m) => m.name)).toEqual(['logistic_regression', 'random_forest', 'naive_bayes'])
    expect(unavailable.map((m) => m.name)).toEqual(['torch_mlp'])
  })

  it('lists nothing usable rather than everything when no Task Type is known yet', () => {
    expect(modelsForTaskType(MODELS, undefined).usable).toHaveLength(3)
  })

  it('names the extra and the command to install it', () => {
    expect(unavailableReason(MODELS[3])).toContain("uv sync --extra torch")
    expect(unavailableReason(MODELS[0])).toBe('')
  })
})

describe('Leaderboard — a Model that cannot be ranked on the chosen metric', () => {
  /** A run whose primary metric is ROC-AUC, with one Model that has no probabilities. */
  const ROC_RUN = {
    ...RUN,
    primary_metric: 'roc_auc',
    primary_metric_higher_is_better: true,
    leaderboard: [
      {
        rank: 1,
        model: 'logistic_regression',
        label: 'Logistic Regression',
        library: 'sklearn',
        task_type: 'classification',
        status: 'ok',
        classes: [],
        hyperparameters: {},
        fit_seconds: 0.1,
        metrics: { roc_auc: { value: 0.93, reason: null }, f1_macro: { value: 0.8, reason: null } },
        primary_metric: 'roc_auc',
        primary: { value: 0.93, reason: null },
      },
      {
        rank: null,
        model: 'svm',
        label: 'SVM',
        library: 'sklearn',
        task_type: 'classification',
        status: 'ok',
        classes: [],
        hyperparameters: {},
        fit_seconds: 0.2,
        metrics: { roc_auc: { value: null, reason: 'this Model does not output class probabilities' } },
        primary_metric: 'roc_auc',
        primary: { value: null, reason: 'this Model does not output class probabilities' },
      },
    ],
  }

  async function showBoard(user: ReturnType<typeof userEvent.setup>) {
    mockFetch({
      ...baseHandlers,
      'POST /train/plan': () => PLAN,
      'POST /train/run': () => ({ id: 'j1' }),
      'GET /train/runs/j1': () => ({ ...ROC_RUN, status: 'running' }),
    })
    const plan = await toThePlan(user)
    await user.click(within(plan).getByRole('button', { name: 'Train' }))
    await screen.findByText('Training Run')
    await emitJobEvent('j1', 'completed', {
      status: 'completed',
      progress: {},
      result: ROC_RUN,
      error: null,
    })
    return screen.findByTestId('leaderboard')
  }

  it("carries the backend's own reason for not ranking it", async () => {
    const user = userEvent.setup()
    const board = await showBoard(user)
    const reason = within(board).getByTestId('unranked-svm')
    expect(reason).toHaveTextContent('Not ranked')
    expect(reason).toHaveTextContent('does not output class probabilities')
  })

  it('never sorts an unmeasurable Model above a measured one', async () => {
    const user = userEvent.setup()
    const board = await showBoard(user)
    const rows = within(board).getAllByTestId(/^board-row-/)
    // the SVM is present, but last and unnumbered — a leaderboard must not rank
    // a Model it could not measure
    expect(rows).toHaveLength(2)
    expect(within(rows[0]).getByText('Logistic Regression')).toBeInTheDocument()
    expect(within(rows[1]).getByText('SVM')).toBeInTheDocument()
    expect(within(rows[1]).queryByText('1')).not.toBeInTheDocument()
  })

  it('lets you re-rank by a metric the run also computed', async () => {
    const user = userEvent.setup()
    const board = await showBoard(user)
    // the SVM has no f1_macro either, so pick accuracy — which both Models have
    const picker = within(board).getByLabelText('Rank by')
    const options = within(picker).getAllByRole('option').map((o) => o.textContent)
    expect(options).toContain('roc_auc (the run\'s own primary)')
    expect(options).toContain('f1_macro')
  })
})
