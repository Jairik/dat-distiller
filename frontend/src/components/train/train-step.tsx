/**
 * The Train step: pick a Target, choose the Models, see what will happen, run.
 *
 * The step is built around the **Plan**. Choosing a Target and ticking Models
 * does not start anything; it asks the backend what it *would* do and shows the
 * answer — the Task Type, the split sizes, which Models are usable and why the
 * others are not, the metrics that will be computed. A request that would be
 * refused comes back as a sentence you can read, before you click Train, rather
 * than a 422 after.
 *
 * The split, the seed and the preprocessing are the backend's business and are
 * shown read-only. The step never invents a test fraction: that is how a run and
 * its plan can drift apart.
 */

import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { motion, useReducedMotion } from 'motion/react'
import { TriangleAlertIcon } from 'lucide-react'

import { CardButton } from '@/components/cards/card-viewer'
import { ChecksPanel } from '@/components/checks/checks-panel'
import { JobProgress } from '@/components/jobs/job-progress'
import { TrainingRunsPanel } from '@/components/cards/training-runs'
import { LeaderboardPanel } from '@/components/train/leaderboard'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Checkbox } from '@/components/ui/checkbox'
import { Label } from '@/components/ui/label'
import { apiGet } from '@/api/client'
import { fmtNumber } from '@/lib/format'
import {
  modelsForTaskType,
  unavailableReason,
  useMetrics,
  useModels,
  usePlan,
  useStartTraining,
  type MetricSpecDto,
  type ModelSpecDto,
  type TrainBody,
  type TrainPlan,
  type TrainingRun,
} from '@/lib/train'
import { useProjectContext } from '@/routes/project-page'

export function TrainStep() {
  const { project, versions } = useProjectContext()
  const navigate = useNavigate()
  const [params] = useSearchParams()

  const versionId = params.get('version') ?? params.get('v') ?? versions.at(-1)?.id ?? ''

  const [target, setTarget] = useState<string | null>(null)
  const [sensitive, setSensitive] = useState<string>('')
  const [excluded, setExcluded] = useState<string[]>([])
  const [chosen, setChosen] = useState<string[]>([])
  const [primary, setPrimary] = useState<string>('')
  const [tune, setTune] = useState(false)
  const [plan, setPlan] = useState<TrainPlan | null>(null)
  const [jobId, setJobId] = useState<string | null>(null)
  const [settledRun, setSettledRun] = useState<TrainingRun | null>(null)

  const models = useModels()
  const planCall = usePlan()
  const start = useStartTraining()

  // the Target is the only thing that tells us the Task Type; until one is
  // chosen we cannot say which Models apply, so we do not pretend to
  const columns = useColumns(versionId)
  const labelFamilies = useLabelFamilies(columns)
  const taskType = plan?.task_type
  const { usable, unavailable } = modelsForTaskType(models.data?.models ?? [], taskType)
  const metrics = useMetrics(taskType)

  const body: TrainBody | null =
    versionId && target
      ? {
          version_id: versionId,
          target,
          sensitive_attribute: sensitive || null,
          exclude_features: excluded,
          models: chosen.length > 0 ? chosen : null,
          primary_metric: primary || null,
          tuning: tune
            ? { enabled: true, strategy: 'random', n_iter: 10, cv: 3 }
            : { enabled: false, strategy: 'random', n_iter: 10, cv: 3 },
        }
      : null

  const metricsForTask: MetricSpecDto[] = metrics.data?.task_types[taskType ?? '']?.metrics ?? []

  return (
    <div className="flex flex-col gap-6">
      <Card>
        <CardHeader>
          <CardTitle>Train</CardTitle>
          <CardDescription>
            Pick a Target, choose the Models, and check the plan before spending anything. The
            held-out test split and the seed are derived by the backend, not chosen here.
          </CardDescription>
        </CardHeader>

        <CardContent className="flex flex-col gap-5">
          {versions.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              No Dataset Versions yet. Generate, upload and Label one first.
            </p>
          ) : (
            <>
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="train-version">Dataset Version</Label>
                <select
                  id="train-version"
                  className={SELECT_CLASS}
                  value={versionId}
                  onChange={() => {
                    setTarget(null)
                    setChosen([])
                    setPlan(null)
                    setJobId(null)
                  }}
                >
                  {versions.map((v) => (
                    <option key={v.id} value={v.id}>
                      {`v${v.number} · ${fmtNumber(v.row_count)} rows · ${v.origin}`}
                    </option>
                  ))}
                </select>
              </div>

              <fieldset className="flex flex-col gap-2">
                <legend className="text-sm font-medium">Target</legend>
                <p className="text-xs text-muted-foreground">
                  The Label Column to predict. Its values decide the Task Type — and the Task Type
                  decides which Models can be trained at all.
                </p>
                {labelFamilies.length === 0 ? (
                  <p className="text-sm text-muted-foreground">
                    {versionId
                      ? 'Reading the Label Columns of this version…'
                      : 'Choose a Dataset Version.'}
                  </p>
                ) : (
                  <ul className="grid gap-1 sm:grid-cols-2">
                    {labelFamilies.map((family) => (
                      <li key={family} className="flex items-center gap-2 text-sm">
                        <Checkbox
                          id={`target-${family}`}
                          checked={target === family}
                          onCheckedChange={(checked) => {
                            if (checked !== true) return
                            setTarget(family)
                            setChosen([])
                            setPlan(null)
                            setPrimary('')
                          }}
                        />
                        <Label htmlFor={`target-${family}`} className="font-mono font-normal">
                          {family}
                        </Label>
                      </li>
                    ))}
                  </ul>
                )}
              </fieldset>
            </>
          )}

          {target && (
            <>
              <SensitiveAttribute
                columns={columns}
                target={target}
                value={sensitive}
                onChange={(next) => {
                  setSensitive(next)
                  setPlan(null)
                }}
                excluded={excluded}
                onToggleExcluded={(column) => {
                  setExcluded((current) =>
                    current.includes(column)
                      ? current.filter((c) => c !== column)
                      : [...current, column],
                  )
                  setPlan(null)
                }}
              />

              <ModelPicker
                models={usable}
                unavailable={unavailable}
                chosen={chosen}
                onChange={(next) => {
                  setChosen(next)
                  setPlan(null)
                }}
              />

              {taskType && metricsForTask.length > 0 && (
                <PrimaryMetricPicker
                  metrics={metricsForTask}
                  value={primary}
                  defaultMetric={
                    metrics.data?.task_types[taskType]?.default_primary_metric ?? ''
                  }
                  onChange={(next) => {
                    setPrimary(next)
                    setPlan(null)
                  }}
                />
              )}

              <label className="flex items-start gap-2 text-sm">
                <Checkbox
                  checked={tune}
                  onCheckedChange={(checked) => {
                    setTune(checked === true)
                    setPlan(null)
                  }}
                />
                <span>
                  Tune hyperparameters
                  <span className="block text-xs text-muted-foreground">
                    A random search with cross-validation, on the training split only. Slower, and
                    the held-out test rows are never seen by the search.
                  </span>
                </span>
              </label>
            </>
          )}
        </CardContent>

        <CardFooter className="flex flex-wrap items-center gap-3">
          <Button
            disabled={!body || planCall.isPending}
            onClick={() => {
              if (body) planCall.mutate(body, { onSuccess: (p) => setPlan(p) })
            }}
          >
            {planCall.isPending ? 'Planning…' : 'See the plan'}
          </Button>
          {body && !plan && (
            <p className="text-sm text-muted-foreground">
              Check the plan before training — it is free, and it is where a bad Target shows up.
            </p>
          )}
        </CardFooter>
      </Card>

      {planCall.isError && (
        <p role="alert" className="flex items-start gap-2 text-sm text-destructive">
          <TriangleAlertIcon aria-hidden className="mt-0.5 size-4 shrink-0" />
          {(planCall.error as Error).message}
        </p>
      )}

      {plan && (
        <PlanPanel
          plan={plan}
          onTrain={() => {
            if (!body) return
            setJobId(null)
            start.mutate(body, { onSuccess: setJobId })
          }}
          canTrain={!start.isPending && !jobId}
        />
      )}

      {start.isError && (
        <p role="alert" className="text-sm text-destructive">
          {(start.error as Error).message}
        </p>
      )}

      {jobId && (
        <TrainingRunPanel
          runId={jobId}
          onSettled={setSettledRun}
          onAgain={() => {
            setJobId(null)
            setPlan(null)
            setSettledRun(null)
          }}
        />
      )}

      {jobId && settledRun && (
        <>
          <div className="flex flex-wrap gap-2">
            <CardButton
              doc={{
                kind: 'model',
                id: jobId,
                title: `Model Card — ${settledRun.target ?? 'Training Run'}`,
                subtitle: `seed ${settledRun.seed ?? '—'}`,
              }}
            />
          </div>
          <ChecksPanel
            subjectType="training_run"
            subjectId={jobId}
            continueLabel="Continue"
            onContinue={() => navigate(`/projects/${project.id}/train?run=${jobId}`)}
          />
        </>
      )}

      <TrainingRunsPanel />
    </div>
  )
}

const SELECT_CLASS =
  'h-9 w-full min-w-0 rounded-md border border-input bg-transparent px-3 py-1 text-base shadow-xs outline-none transition-[color,box-shadow] focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 disabled:pointer-events-none disabled:opacity-50 md:text-sm dark:bg-input/30'

// -- the pieces ---------------------------------------------------------------

function SensitiveAttribute({
  columns,
  target,
  value,
  onChange,
  excluded,
  onToggleExcluded,
}: {
  columns: string[]
  target: string
  value: string
  onChange: (next: string) => void
  excluded: string[]
  onToggleExcluded: (column: string) => void
}) {
  const candidates = columns.filter((c) => c !== target)
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="sensitive">Sensitive Attribute</Label>
        <select
          id="sensitive"
          className={SELECT_CLASS}
          value={value}
          onChange={(event) => onChange(event.target.value)}
        >
          <option value="">None</option>
          {candidates.map((column) => (
            <option key={column} value={column}>
              {column}
            </option>
          ))}
        </select>
        <p className="text-xs text-muted-foreground">
          The column a Fairness Report would be computed against. It is left out of the features
          by default — a model that can see the attribute cannot be measured on it fairly.
        </p>
      </div>

      <details className="text-sm">
        <summary className="cursor-pointer text-muted-foreground">
          Exclude features ({excluded.length})
        </summary>
        <ul className="mt-2 grid gap-1 sm:grid-cols-2">
          {candidates.map((column) => (
            <li key={column} className="flex items-center gap-2">
              <Checkbox
                id={`exclude-${column}`}
                checked={excluded.includes(column)}
                onCheckedChange={() => onToggleExcluded(column)}
              />
              <Label htmlFor={`exclude-${column}`} className="font-mono font-normal">
                {column}
              </Label>
            </li>
          ))}
        </ul>
      </details>
    </div>
  )
}

function ModelPicker({
  models,
  unavailable,
  chosen,
  onChange,
}: {
  models: ModelSpecDto[]
  unavailable: ModelSpecDto[]
  chosen: string[]
  onChange: (next: string[]) => void
}) {
  if (models.length === 0 && unavailable.length === 0) {
    return <p className="text-sm text-muted-foreground">Reading the Model registry…</p>
  }
  return (
    <fieldset className="flex flex-col gap-2">
      <legend className="text-sm font-medium">Models</legend>
      <div className="flex flex-wrap items-center gap-2">
        <Button
          size="sm"
          variant="ghost"
          disabled={models.length === 0}
          onClick={() => onChange(chosen.length === models.length ? [] : models.map((m) => m.name))}
        >
          {chosen.length === models.length ? 'Clear all' : 'Select every Model'}
        </Button>
        <span className="text-xs text-muted-foreground">
          {chosen.length === 0
            ? 'None ticked means every Model for this Task Type.'
            : `${chosen.length} of ${models.length} selected`}
        </span>
      </div>

      <ul className="flex flex-col gap-1">
        {models.map((model) => (
          <li key={model.name} className="flex items-start gap-2 text-sm">
            <Checkbox
              id={`model-${model.name}`}
              checked={chosen.includes(model.name)}
              onCheckedChange={(checked) =>
                onChange(
                  checked === true
                    ? [...chosen, model.name]
                    : chosen.filter((name) => name !== model.name),
                )
              }
            />
            <span className="flex flex-col">
              <Label htmlFor={`model-${model.name}`} className="font-normal">
                {model.label}
              </Label>
              <span className="text-xs text-muted-foreground">
                {model.library}
                {model.notes ? ` — ${model.notes}` : ''}
              </span>
            </span>
          </li>
        ))}
      </ul>

      {unavailable.length > 0 && (
        <ul className="flex flex-col gap-1 text-xs text-amber-400">
          {unavailable.map((model) => (
            <li key={model.name}>
              {`${model.label} — ${unavailableReason(model)}`}
            </li>
          ))}
        </ul>
      )}
    </fieldset>
  )
}

function PrimaryMetricPicker({
  metrics,
  value,
  defaultMetric,
  onChange,
}: {
  metrics: MetricSpecDto[]
  value: string
  defaultMetric: string
  onChange: (next: string) => void
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <Label htmlFor="primary-metric">Rank by</Label>
      <select
        id="primary-metric"
        className={SELECT_CLASS}
        value={value}
        onChange={(event) => onChange(event.target.value)}
      >
        <option value="">{`Default (${defaultMetric})`}</option>
        {metrics.map((metric) => (
          <option key={metric.name} value={metric.name}>
            {`${metric.label} (${metric.higher_is_better ? 'higher' : 'lower'} is better)`}
          </option>
        ))}
      </select>
      <p className="text-xs text-muted-foreground">
        {value
          ? (metrics.find((m) => m.name === value)?.description ?? '')
          : 'The Task Type default. Every metric is still computed and shown; this only decides the order.'}
      </p>
    </div>
  )
}

function PlanPanel({
  plan,
  onTrain,
  canTrain,
}: {
  plan: TrainPlan
  onTrain: () => void
  canTrain: boolean
}) {
  const reduced = useReducedMotion()
  return (
    <motion.div
      initial={reduced ? false : { opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: reduced ? 0 : 0.24 }}
    >
      <Card data-testid="train-plan">
        <CardHeader>
          <CardTitle>The plan</CardTitle>
          <CardDescription>
            What training this Target would actually do. Nothing has run yet.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          <dl className="grid gap-2 text-sm sm:grid-cols-2">
            <Fact label="Task Type" value={plan.task_type} />
            <Fact label="Label Column" value={plan.label_family} />
            <Fact
              label="Rows"
              value={`${fmtNumber(plan.kept_rows)} kept of ${fmtNumber(plan.rows)}`}
            />
            <Fact
              label="Split"
              value={`${fmtNumber(plan.train_rows)} train / ${fmtNumber(plan.test_rows)} held out`}
            />
            <Fact label="Features" value={fmtNumber(plan.n_source_features)} />
            <Fact label="Seed" value={String(plan.seed)} />
            <Fact
              label="Rank by"
              value={`${plan.primary_metric_label} (${plan.primary_metric_higher_is_better ? 'higher' : 'lower'} is better)`}
            />
            <Fact
              label="Tuning"
              value={
                plan.tuning.enabled
                  ? `${plan.tuning.strategy} search, ${plan.tuning.n_iter} candidates, ${plan.tuning.cv}-fold CV on the training split`
                  : 'off — defaults for every Model'
              }
            />
          </dl>

          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-xs text-muted-foreground">Metrics that will be computed:</span>
            {plan.metrics_available.map((metric) => (
              <Badge key={metric} variant="outline" className="text-xs">
                {metric}
              </Badge>
            ))}
          </div>

          {plan.unavailable_models.length > 0 && (
            <p className="text-xs text-amber-400">
              {`Not runnable here: ${plan.unavailable_models
                .map((m) => `${m.model} (${m.reason})`)
                .join(', ')}`}
            </p>
          )}
        </CardContent>
        <CardFooter>
          <Button disabled={!canTrain} onClick={onTrain}>
            {canTrain ? 'Train' : 'Training…'}
          </Button>
        </CardFooter>
      </Card>
    </motion.div>
  )
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col">
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className="font-medium">{value}</dd>
    </div>
  )
}

function TrainingRunPanel({
  runId,
  onAgain,
  onSettled,
}: {
  runId: string
  onAgain: () => void
  onSettled: (run: TrainingRun) => void
}) {
  // the run payload comes from the job's own result, so there is no second
  // query to keep in step with the job's status
  return (
    <Card>
      <CardHeader>
        <CardTitle>Training Run</CardTitle>
        <CardDescription>
          Each Model is fitted, then scored once on the held-out test split. The run checkpoints as
          it goes.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <JobProgress
          jobId={runId}
          label="Training"
          onSettled={(result) => result && onSettled(result as unknown as TrainingRun)}
          renderResult={(result) => (
            <div className="flex flex-col gap-4">
              <RunWarnings run={result as unknown as TrainingRun} />
              <Leaderboard run={result as unknown as TrainingRun} />
            </div>
          )}
          terminalExtra={
            <Button size="sm" variant="ghost" onClick={onAgain}>
              Train again
            </Button>
          }
        />
      </CardContent>
    </Card>
  )
}

/**
 * The columns a Fairness Report could be run against.
 *
 * The Target is excluded: a group defined by the thing you are predicting is not
 * a group, and the backend refuses it. Everything else the run saw is offered.
 */
function fairnessColumns(run: TrainingRun | null): string[] {
  const setup = run?.setup ?? {}
  const target = String(setup.target ?? run?.target ?? '')
  // `excluded_columns` is a column -> reason mapping, so its keys are the names
  const excluded = setup.excluded_columns
  const excludedNames =
    excluded && !Array.isArray(excluded) ? Object.keys(excluded) : (excluded as string[] | undefined) ?? []
  const known = [
    ...((setup.feature_columns as string[] | undefined) ?? []),
    ...excludedNames,
  ]
  return [...new Set(known)].filter((column) => column !== target)
}

/** What the run itself wants to say — a property of the run, not of the board. */
function RunWarnings({ run }: { run: TrainingRun | null }) {
  const warnings = run?.warnings ?? []
  if (warnings.length === 0) return null
  return (
    <ul className="flex flex-col gap-0.5 text-xs text-amber-400" data-testid="run-warnings">
      {warnings.map((warning) => (
        <li key={warning}>{warning}</li>
      ))}
    </ul>
  )
}

function Leaderboard({ run }: { run: TrainingRun | null }) {
  if (!run?.leaderboard?.length) {
    return <p className="text-sm text-muted-foreground">No leaderboard yet.</p>
  }
  return (
    <LeaderboardPanel
      runId={String(run.training_run_id ?? '')}
      board={run.leaderboard}
      taskType={String(run.task_type ?? '')}
      primaryMetric={String(run.primary_metric ?? 'f1_macro')}
      primaryHigherIsBetter={run.primary_metric_higher_is_better ?? true}
      columns={fairnessColumns(run)}
      declaredSensitiveAttribute={(run.setup?.sensitive_attribute as string) ?? null}
    />
  )
}

// -- column discovery ---------------------------------------------------------

/** This version's column names, from a one-row preview. */
function useColumns(versionId: string | null): string[] {
  const query = useQuery({
    queryKey: ['train-columns', versionId],
    queryFn: () =>
      apiGet<{ columns: Array<{ name: string }> }>(
        `/dataset-versions/${versionId}/preview?page_size=1`,
      ).then((body) => body.columns.map((c) => c.name)),
    enabled: Boolean(versionId),
    retry: false,
  })
  return query.data ?? []
}

/**
 * The Label Columns you can pick as a Target.
 *
 * A Label Column is a family name, never one of the siblings Labeling adds
 * alongside it (`<name>__confidence`, `<name>__p_<option>`, …). Offering
 * `is_churn__confidence` as a Target would train a model to predict Jev's own
 * certainty, so the siblings are filtered out here rather than discovered when
 * the model scores badly.
 */
function useLabelFamilies(columns: string[]): string[] {
  return useMemo(
    () => columns.filter((c) => !c.startsWith('__') && !/__/.test(c)),
    [columns],
  )
}
