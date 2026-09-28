/**
 * The Training Run leaderboard, and everything you can do with a row.
 *
 * Three decisions shape this:
 *
 * **Nothing is computed until you ask.** Each Model's plot buttons start closed,
 * and closing the panel throws the result away rather than showing it again for
 * free. A leaderboard that silently refits eight models to draw charts nobody
 * looked at is a leaderboard you wait for.
 *
 * **A Model that could not be ranked says why, in the row.** It is not hidden,
 * and it is never sorted above a Model with a real number — sorting an
 * unmeasurable Model first is the one thing a leaderboard must never do.
 *
 * **Downloads are plain links.** A bundle or a predictions file is a file; making
 * it a click on an anchor means it works with the browser's own download
 * handling, and a test can assert on the href rather than on a mock.
 */

import { useState } from 'react'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import { DownloadIcon, LineChartIcon, LoaderIcon, ScaleIcon, UploadIcon } from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import {
  ALL_PLOTS,
  CLASSIFICATION_PLOTS,
  PLOT_LABELS,
  REGRESSION_PLOTS,
  bundleUrl,
  metricAvailable,
  rankBy,
  rankableMetrics,
  usePredict,
  usePlot,
  type LeaderboardEntry,
  type PlotName,
  type RankedEntry,
  type PlotResult,
} from '@/lib/evaluate'
import {
  ConfusionMatrixView,
  FeatureImportanceView,
  PlotCard,
  PrecisionRecallView,
  ResidualsView,
  RocView,
} from '@/components/train/plots'
import { fmtNumber } from '@/lib/format'
import { FairnessPanel } from '@/components/fairness/fairness-panel'

export function LeaderboardPanel({
  runId,
  board,
  taskType,
  primaryMetric,
  primaryHigherIsBetter,
  columns = [],
  declaredSensitiveAttribute = null,
}: {
  runId: string
  board: LeaderboardEntry[]
  taskType: string
  primaryMetric: string
  primaryHigherIsBetter: boolean
  /** Candidate Sensitive Attributes, from the Training Run's own setup. */
  columns?: string[]
  declaredSensitiveAttribute?: string | null
}) {
  const [metric, setMetric] = useState(primaryMetric)
  const [showAll, setShowAll] = useState(false)

  const available = rankableMetrics(board)
  const chosen = available.includes(metric) ? metric : primaryMetric
  const higherIsBetter = metricDirection(board, chosen, primaryMetric, primaryHigherIsBetter)
  const ranked = rankBy(board, chosen, higherIsBetter)

  const visible = showAll ? ranked : ranked.filter((entry) => entry.status === 'ok')

  return (
    <Card data-testid="leaderboard">
      <CardHeader>
        <CardTitle>Leaderboard</CardTitle>
        <CardDescription>
          Every Model that fitted, scored once on the held-out test split. Every metric is computed;
          this only decides the order.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div className="flex min-w-48 flex-col gap-1.5">
            <Label htmlFor="rank-by" className="text-xs text-muted-foreground">
              Rank by
            </Label>
            <select
              id="rank-by"
              className={SELECT_CLASS}
              value={chosen}
              onChange={(event) => setMetric(event.target.value)}
            >
              {available.map((name) => (
                <option key={name} value={name}>
                  {name === primaryMetric ? `${name} (the run's own primary)` : name}
                </option>
              ))}
            </select>
          </div>
          <p className="text-xs text-muted-foreground">
            {`${ranked.filter((e) => e.rankable).length} of ${board.length} rankable on ${chosen} (${higherIsBetter ? 'higher' : 'lower'} is better)`}
          </p>
        </div>

        <ol className="flex flex-col gap-3">
          {visible.map((entry) => (
            <LeaderboardRow
              key={entry.model}
              runId={runId}
              entry={entry}
              metric={chosen}
              higherIsBetter={higherIsBetter}
              taskType={taskType}
              columns={columns}
              declaredSensitiveAttribute={declaredSensitiveAttribute}
            />
          ))}
        </ol>

        {visible.length < board.length && (
          <Button size="sm" variant="ghost" onClick={() => setShowAll(true)}>
            {`Show the ${board.length - visible.length} that did not fit`}
          </Button>
        )}
      </CardContent>
    </Card>
  )
}

const SELECT_CLASS =
  'h-9 w-full min-w-0 rounded-md border border-input bg-transparent px-3 py-1 text-base shadow-xs outline-none transition-[color,box-shadow] focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 disabled:pointer-events-none disabled:opacity-50 md:text-sm dark:bg-input/30'

/** Is a bigger number better for this metric? Read it off the board itself. */
function metricDirection(
  board: LeaderboardEntry[],
  metric: string,
  primary: string,
  primaryHigherIsBetter: boolean,
): boolean {
  if (metric === primary) return primaryHigherIsBetter
  // Compare two Models: whichever scored higher on the backend's own primary
  // should score higher here too. With a one-Model board there is nothing to
  // compare, and higher-is-better is the friendlier default for these metrics.
  const scored = board.filter((e) => e.status === 'ok' && e.metrics?.[metric]?.value != null)
  if (scored.length < 2) return true
  const [a, b] = scored
  return (a.metrics[primary]?.value ?? 0) >= (b.metrics[primary]?.value ?? 0)
}

function LeaderboardRow({
  runId,
  entry,
  metric,
  higherIsBetter,
  taskType,
  columns,
  declaredSensitiveAttribute,
}: {
  runId: string
  entry: RankedEntry
  metric: string
  higherIsBetter: boolean
  taskType: string
  columns: string[]
  declaredSensitiveAttribute: string | null
}) {
  const [open, setOpen] = useState<PlotName | null>(null)
  const [fairness, setFairness] = useState(false)
  const plots: PlotName[] =
    taskType === 'regression'
      ? [...REGRESSION_PLOTS, 'feature_importance']
      : [...CLASSIFICATION_PLOTS, 'feature_importance']
  const value = metric === entry.primary_metric ? entry.primary?.value : entry.metrics?.[metric]?.value

  return (
    <motion.li
      layout
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.22 }}
      data-testid={`board-row-${entry.model}`}
      className={`flex flex-col gap-2 rounded-lg border p-3 ${
        entry.rankable ? 'border-border' : 'border-amber-400/30 bg-amber-400/5'
      }`}
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex min-w-0 flex-col gap-1">
          <span className="flex flex-wrap items-center gap-2 text-sm font-medium">
            <span className="tabular-nums text-muted-foreground">{entry.displayRank ?? '—'}</span>
            {entry.label}
            <Badge variant="outline" className="text-xs">
              {entry.library}
            </Badge>
          </span>
          <span className="text-xs text-muted-foreground">
            {`${metric} ${value === null || value === undefined ? '—' : value.toFixed(4)}${
              higherIsBetter ? ' ↑' : ' ↓'
            }`}
          </span>
          {!entry.rankable && (
            <span className="text-xs text-amber-400" data-testid={`unranked-${entry.model}`}>
              {`Not ranked: ${entry.unrankedReason}`}
            </span>
          )}
        </div>
        <MetricChips entry={entry} metric={metric} />
      </div>

      {entry.status === 'ok' && (
        <div className="flex flex-wrap items-center gap-1.5">
          {plots.map((plot) => {
            const isOpen = open === plot
            return (
              <Button
                key={plot}
                size="sm"
                variant={isOpen ? 'secondary' : 'outline'}
                onClick={() => setOpen(isOpen ? null : plot)}
                aria-expanded={isOpen}
              >
                <LineChartIcon aria-hidden />
                {PLOT_LABELS[plot]}
              </Button>
            )
          })}
          <PredictButton runId={runId} entry={entry} />
          <Button
            size="sm"
            variant={fairness ? 'secondary' : 'outline'}
            onClick={() => setFairness(!fairness)}
            aria-expanded={fairness}
          >
            <ScaleIcon aria-hidden />
            Fairness Report
          </Button>
          <Button size="sm" variant="ghost" asChild>
            <a href={bundleUrl(runId, entry.model)} download={`${entry.model}-bundle.zip`}>
              <DownloadIcon aria-hidden />
              Model Bundle
            </a>
          </Button>
        </div>
      )}

      <AnimatePresence initial={false}>
        {fairness && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <FairnessPanel
              runId={runId}
              columns={columns}
              models={[{ name: entry.model, label: entry.label }]}
              declaredAttribute={declaredSensitiveAttribute}
              defaultModel={entry.model}
            />
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            key={open}
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <OnePlot runId={runId} entry={entry} plot={open} />
          </motion.div>
        )}
      </AnimatePresence>
    </motion.li>
  )
}

function MetricChips({ entry, metric }: { entry: LeaderboardEntry; metric: string }) {
  const others = Object.entries(entry.metrics ?? {}).filter(
    ([name, value]) => name !== metric && name !== 'confusion_matrix' && value?.value != null,
  )
  if (others.length === 0) return null
  return (
    <ul className="flex flex-wrap justify-end gap-1">
      {others.map(([name, value]) => (
        <li key={name}>
          <Badge variant="outline" className="font-mono text-[10px]">
            {`${name} ${value.value?.toFixed(3)}`}
          </Badge>
        </li>
      ))}
    </ul>
  )
}

/** One plot, fetched only because the user opened this panel. */
function OnePlot({
  runId,
  entry,
  plot,
}: {
  runId: string
  entry: LeaderboardEntry
  plot: PlotName
}) {
  const query = usePlot(runId, entry.model, plot, true)

  if (query.isPending) {
    return (
      <p className="flex items-center gap-2 py-4 text-sm text-muted-foreground">
        <LoaderIcon aria-hidden className="size-4 animate-spin" />
        {`Computing the ${PLOT_LABELS[plot].toLowerCase()}…`}
      </p>
    )
  }
  if (query.isError) {
    return (
      <p role="alert" className="py-4 text-sm text-destructive">
        {(query.error as Error).message}
      </p>
    )
  }
  const data = query.data as (PlotResult & { error?: string | null }) | undefined
  if (!data) return null
  // the backend answers 200 with a reason for a plot this Model cannot produce
  if (data.error) {
    return (
      <p className="py-4 text-sm text-amber-400" data-testid="plot-unavailable">
        {data.error}
      </p>
    )
  }

  return (
    <PlotCard plot={plot} title={PLOT_LABELS[plot]}>
      {plot === 'confusion_matrix' && <ConfusionMatrixView data={data as never} />}
      {plot === 'roc' && <RocView data={data as never} />}
      {plot === 'precision_recall' && <PrecisionRecallView data={data as never} />}
      {plot === 'residuals' && <ResidualsView data={data as never} />}
      {plot === 'feature_importance' && <FeatureImportanceView data={data as never} />}
    </PlotCard>
  )
}

function PredictButton({ runId, entry }: { runId: string; entry: RankedEntry }) {
  const [open, setOpen] = useState(false)
  const predict = usePredict()
  const reduced = useReducedMotion()

  return (
    <div className="flex flex-col gap-2">
      <Button size="sm" variant={open ? 'secondary' : 'outline'} onClick={() => setOpen(!open)} aria-expanded={open}>
        <UploadIcon aria-hidden />
        Predict on a CSV
      </Button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={reduced ? false : { opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={reduced ? undefined : { opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="flex flex-col gap-2 rounded-md border border-border p-2">
              <Label htmlFor={`predict-${entry.model}`} className="text-xs text-muted-foreground">
                {`A CSV with the ${entry.model} Model's feature columns`}
              </Label>
              <input
                id={`predict-${entry.model}`}
                type="file"
                accept=".csv,text/csv"
                className="text-xs"
                onChange={(event) => {
                  const file = event.target.files?.[0]
                  if (file) predict.mutate({ runId, model: entry.model, file })
                }}
              />
              {predict.isPending && (
                <p className="text-xs text-muted-foreground">
                  {'Predicting… this refits the Model, so it takes a moment.'}
                </p>
              )}
              {predict.isError && (
                <p role="alert" className="text-xs text-destructive">
                  {(predict.error as Error).message}
                </p>
              )}
              {predict.data && (
                <div className="flex flex-col gap-1" data-testid={`predict-result-${entry.model}`}>
                  <p className="text-xs">
                    {`${fmtNumber(predict.data.rows)} row(s) predicted. Columns: ${predict.data.columns
                      .slice(-3)
                      .join(', ')}${predict.data.columns.length > 3 ? '…' : ''}`}
                  </p>
                  <Button size="sm" variant="outline" asChild>
                    <a href={predict.data.download_url} download="predictions.csv">
                      <DownloadIcon aria-hidden />
                      Download predictions
                    </a>
                  </Button>
                  <details>
                    <summary className="cursor-pointer text-xs text-muted-foreground">
                      See the first rows
                    </summary>
                    <pre className="mt-1 overflow-x-auto text-[10px]">
                      {JSON.stringify(predict.data.predictions.slice(0, 3), null, 2)}
                    </pre>
                  </details>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export { ALL_PLOTS, metricAvailable }
