/**
 * The Fairness Report view.
 *
 * This panel is about one thing: making a fairness finding *readable at a glance*
 * without turning it into something alarming or something reassuring. So:
 *
 * - **The bars are the headline.** Every group, side by side, per metric, with a
 *   marker where the threshold sits. A reader should be able to see "this group
 *   is much lower" before reading a word.
 * - **The numbers are always there too**, because a bar you cannot check is a
 *   claim, not a measurement.
 * - **A group that could not be measured is shown, greyed, with its reason.**
 *   Dropping it would quietly turn a three-group report into a two-group one and
 *   read as "these two are treated equally".
 * - **Flagged gaps route through the Checks panel**, so acknowledging one is the
 *   same act as acknowledging any other warning in this app.
 */

import { useState } from 'react'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { ScaleIcon, TriangleAlertIcon } from 'lucide-react'

import { ChecksPanel } from '@/components/checks/checks-panel'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  METRIC_LABELS,
  METRIC_PLAIN,
  headlineGap,
  explainGap,
  useFairnessReport,
  type FairnessGroup,
  type FairnessReport,
} from '@/lib/fairness'

const SELECT_CLASS =
  'h-9 w-full min-w-0 rounded-md border border-input bg-transparent px-3 py-1 text-base shadow-xs outline-none transition-[color,box-shadow] focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 disabled:pointer-events-none disabled:opacity-50 md:text-sm dark:bg-input/30'

export function FairnessPanel({
  runId,
  columns,
  models,
  declaredAttribute,
  defaultModel,
  onReport,
}: {
  runId: string
  columns: string[]
  models: Array<{ name: string; label: string }>
  declaredAttribute: string | null
  defaultModel: string
  onReport?: (report: FairnessReport) => void
}) {
  const [attribute, setAttribute] = useState(declaredAttribute ?? '')
  const [model, setModel] = useState(defaultModel)
  const [threshold, setThreshold] = useState('')
  const report = useFairnessReport()
  const reduced = useReducedMotion()

  const run = () => {
    const parsed = threshold.trim() === '' ? undefined : Number(threshold)
    report.mutate(
      {
        runId,
        sensitive_attribute: attribute,
        model,
        gap_threshold: parsed !== undefined && Number.isFinite(parsed) ? parsed : undefined,
      },
      { onSuccess: onReport },
    )
  }

  return (
    <div className="flex flex-col gap-4">
      <Card data-testid="fairness-panel">
        <CardHeader>
          <CardTitle>Fairness Report</CardTitle>
          <CardDescription>
            How one Model treats each group of a Sensitive Attribute, on the held-out test split.
            Nothing is computed until you ask, and a gap over the threshold becomes a Check you
            have to acknowledge.
          </CardDescription>
        </CardHeader>

        <CardContent className="flex flex-col gap-4">
          <div className="grid gap-3 sm:grid-cols-3">
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="fairness-attribute">Report on this Sensitive Attribute</Label>
              <select
                id="fairness-attribute"
                className={SELECT_CLASS}
                value={attribute}
                onChange={(event) => setAttribute(event.target.value)}
              >
                <option value="">Choose a column…</option>
                {columns.map((column) => (
                  <option key={column} value={column}>
                    {column}
                  </option>
                ))}
              </select>
            </div>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="fairness-model">Model</Label>
              <select
                id="fairness-model"
                className={SELECT_CLASS}
                value={model}
                onChange={(event) => setModel(event.target.value)}
              >
                {models.map((m) => (
                  <option key={m.name} value={m.name}>
                    {m.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="fairness-threshold">Gap threshold (optional)</Label>
              <Input
                id="fairness-threshold"
                inputMode="decimal"
                placeholder="the default"
                value={threshold}
                onChange={(event) => setThreshold(event.target.value)}
              />
            </div>
          </div>
          <p className="text-xs text-muted-foreground">
            The Sensitive Attribute is held out of the Model's features by default, so a gap here is
            about how the Model treats a group it cannot identify.
          </p>
        </CardContent>

        <CardFooter>
          <Button disabled={!attribute || report.isPending} onClick={run}>
            <ScaleIcon aria-hidden />
            {report.isPending ? 'Measuring…' : 'Measure this group'}
          </Button>
          {report.data && (
            <Button size="sm" variant="ghost" onClick={run} disabled={report.isPending}>
              Measure again
            </Button>
          )}
        </CardFooter>
      </Card>

      {report.isError && (
        <p role="alert" className="flex items-start gap-2 text-sm text-destructive">
          <TriangleAlertIcon aria-hidden className="mt-0.5 size-4 shrink-0" />
          {(report.error as Error).message}
        </p>
      )}

      <AnimatePresence>
        {report.data && (
          <motion.div
            key="report"
            initial={reduced ? false : { opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: reduced ? 0 : 0.24 }}
            className="flex flex-col gap-4"
          >
            <ReportHeadline report={report.data} />
            <GroupBars report={report.data} />
            <GapTable report={report.data} />
            <UnmeasuredGroups report={report.data} />
            <Notes report={report.data} />
          </motion.div>
        )}
      </AnimatePresence>

      {report.data && report.data.checks_raised.length > 0 && (
        <>
          <p className="text-sm text-amber-400" data-testid="checks-raised-summary">
            {`${report.data.gaps_exceeding_threshold.length} gap(s) over the threshold raised ` +
              `${report.data.checks_raised.length} Check(s) on this Training Run. ` +
              'Acknowledge them below once you have read them.'}
          </p>
          <ChecksPanel
            subjectType="training_run"
            subjectId={runId}
            continueLabel="Continue"
            onContinue={() => undefined}
          />
        </>
      )}
    </div>
  )
}

// -- the parts ----------------------------------------------------------------

function ReportHeadline({ report }: { report: FairnessReport }) {
  const gap = headlineGap(report)
  const flagged = report.gaps_exceeding_threshold.length > 0
  return (
    <Card data-testid="fairness-headline">
      <CardHeader>
        <CardTitle>
          {`${report.model_label} by ${report.sensitive_attribute}`}
        </CardTitle>
        <CardDescription>
          {`${report.n_groups_measured} of ${report.n_groups} group(s) could be measured, over ${report.split.test_rows} held-out row(s).`}
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-2">
        {gap ? (
          <p className={`text-sm ${flagged ? 'text-amber-400' : 'text-muted-foreground'}`}>
            {explainGap(gap, report)}
          </p>
        ) : (
          <p className="text-sm text-muted-foreground">No gap could be measured on this run.</p>
        )}
        <ul className="flex flex-wrap gap-1.5">
          <Badge variant="outline" className="text-xs">
            {`threshold ${report.threshold}`}
          </Badge>
          <Badge variant="outline" className="text-xs">
            {`target ${report.target}`}
          </Badge>
          {report.positive_class && (
            <Badge variant="outline" className="text-xs">
              {`positive class ${report.positive_class}`}
            </Badge>
          )}
          {report.sensitive_attribute_excluded_from_features ? (
            <Badge variant="outline" className="border-emerald-400/40 text-xs text-emerald-400">
              attribute not a feature
            </Badge>
          ) : (
            <Badge variant="outline" className="border-amber-400/40 text-xs text-amber-400">
              the Model could see this attribute
            </Badge>
          )}
        </ul>
      </CardContent>
    </Card>
  )
}

/** Per-group bars for each metric, with the threshold drawn where it sits. */
function GroupBars({ report }: { report: FairnessReport }) {
  const measured = report.groups.filter((g) => g.measured)
  const metrics = [...new Set(measured.flatMap((g) => Object.keys(g.metrics)))].filter(
    (name) => measured.some((g) => g.metrics[name]?.value !== null),
  )
  if (metrics.length === 0) {
    return (
      <Card>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            None of the groups had a value for any metric, so there is nothing to plot.
          </p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card data-testid="fairness-bars">
      <CardHeader>
        <CardTitle>Per-group metrics</CardTitle>
        <CardDescription>
          One bar per group. The dashed line is the gap threshold; a group well clear of it is a
          group the Model treats differently from the rest.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {metrics.map((metric) => (
          <MetricBars
            key={metric}
            metric={metric}
            report={report}
            groups={measured}
            threshold={report.gaps.find((g) => g.metric === metric && g.comparable)?.threshold}
          />
        ))}
      </CardContent>
    </Card>
  )
}

function MetricBars({
  metric,
  report,
  groups,
  threshold,
}: {
  metric: string
  report: FairnessReport
  groups: FairnessGroup[]
  threshold?: number
}) {
  // A group with no value for this metric is left out rather than drawn as a
  // real 0.0. The numbers table below prints an em dash for exactly these groups,
  // so `?? 0` made the chart and the table disagree about the same measurement —
  // and "this group scored nothing" is not what a zero-length bar says.
  const data = groups
    .map((group) => ({ group: group.group, value: group.metrics[metric]?.value ?? null, n: group.n_test }))
    .filter((row): row is { group: string; value: number; n: number } => row.value !== null)

  // `[0, 1]` is right for a rate and wrong for everything else: an MAE of 1.2
  // and an MAE of 3.4 both clamp to a full-width bar, so a regression report's
  // headline chart was a row of identical bars — the opposite of what the panel
  // exists to show. The domain comes from the data, with 0 kept as the baseline
  // because every one of these metrics is a quantity where zero is meaningful.
  const max = data.reduce((widest, row) => Math.max(widest, row.value), 0)
  const domain: [number, number] = [0, max > 0 ? max : 1]

  return (
    <section data-testid={`bars-${metric}`}>
      <p className="text-sm font-medium">{METRIC_LABELS[metric] ?? metric}</p>
      <p className="text-xs text-muted-foreground">{METRIC_PLAIN[metric] ?? ''}</p>
      <div role="img" aria-label={`${METRIC_LABELS[metric] ?? metric} by group`}>
        <ResponsiveContainer width="100%" height={Math.max(120, data.length * 34)}>
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 8, right: 60, bottom: 8, left: 8 }}
          >
            <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
            <XAxis
              type="number"
              domain={domain}
              tickFormatter={(v) => Number(v).toFixed(1)}
            />
            <YAxis type="category" dataKey="group" width={96} />
            <Tooltip formatter={(v: unknown) => Number(v).toFixed(4)} />
            {threshold !== undefined && (
              <ReferenceLine
                x={threshold}
                stroke="var(--color-destructive)"
                strokeDasharray="4 4"
                label={{ value: `gap threshold ${threshold}`, position: 'top', fontSize: 10 }}
              />
            )}
            <Bar dataKey="value" fill="var(--color-primary)" radius={[0, 3, 3, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      {/* the numbers, because a bar you cannot check is a claim */}
      <table className="w-full text-xs" data-testid={`table-${metric}`}>
        <thead>
          <tr>
            <th scope="col" className="px-1 py-0.5 text-left font-medium">
              group
            </th>
            <th scope="col" className="px-1 py-0.5 text-right font-medium">
              held-out rows
            </th>
            <th scope="col" className="px-1 py-0.5 text-right font-medium">
              {METRIC_LABELS[metric] ?? metric}
            </th>
          </tr>
        </thead>
        <tbody>
          {groups.map((group) => (
            <tr key={group.group}>
              <th scope="row" className="px-1 py-0.5 text-left font-medium">
                {group.group}
              </th>
              <td className="px-1 py-0.5 text-right tabular-nums text-muted-foreground">
                {group.n_test}
              </td>
              <td className="px-1 py-0.5 text-right tabular-nums">
                {group.metrics[metric]?.value === null ||
                group.metrics[metric]?.value === undefined
                  ? '—'
                  : group.metrics[metric].value?.toFixed(4)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {report.gaps
        .filter((gap) => gap.metric === metric)
        .map((gap) => (
          <p
            key={gap.name}
            className={`mt-1 text-xs ${gap.exceeds === true ? 'text-amber-400' : 'text-muted-foreground'}`}
          >
            {explainGap(gap, report)}
          </p>
        ))}
    </section>
  )
}

/** Every gap, so a reader sees the ones that were fine as well as the ones that were not. */
function GapTable({ report }: { report: FairnessReport }) {
  return (
    <Card data-testid="fairness-gaps">
      <CardHeader>
        <CardTitle>Gaps</CardTitle>
        <CardDescription>
          Every gap that was measured, widest first. A gap in the Target's own units cannot be
          judged against a rate threshold, and says so rather than being counted as a finding.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ul className="flex flex-col gap-1.5">
          {report.gaps.map((gap) => (
            <li
              key={gap.name}
              data-testid={`gap-${gap.name}`}
              className={`flex flex-wrap items-baseline gap-2 rounded-md border px-2 py-1 text-sm ${
                gap.exceeds === true
                  ? 'border-amber-400/40 bg-amber-400/5'
                  : 'border-border'
              }`}
            >
              <span className="font-medium">{gap.label}</span>
              <span className="tabular-nums">
                {gap.value === null ? '—' : gap.value.toFixed(4)}
                {gap.comparable ? '' : ` ${gap.unit}`}
              </span>
              <span className="text-xs text-muted-foreground">
                {`${gap.groups_compared} of ${gap.groups_total} group(s) compared`}
              </span>
              {gap.exceeds === true && (
                <Badge variant="outline" className="border-amber-400/40 text-xs text-amber-400">
                  over the threshold
                </Badge>
              )}
              {/* "not judged" covers *both* reasons a threshold cannot call a
                  gap a finding: it was not comparable, or there was nothing to
                  compare. Without the badge, a gap in the Target's own units
                  sits in the list looking like one that simply did not trip. */}
              {gap.exceeds === null && (
                <Badge variant="outline" className="text-xs text-muted-foreground">
                  not judged
                </Badge>
              )}
              {gap.is_widest && (
                <Badge variant="outline" className="text-xs">
                  widest
                </Badge>
              )}
              {gap.reason && <span className="text-xs text-muted-foreground">{gap.reason}</span>}
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
  )
}

/**
 * The groups that could not be measured, with the reason.
 *
 * Shown, not dropped: a report that quietly omitted a group reads as "these two
 * are treated equally", which is not what the data says.
 */
function UnmeasuredGroups({ report }: { report: FairnessReport }) {
  if (report.unmeasured_groups.length === 0) return null
  return (
    <Card data-testid="unmeasured-groups">
      <CardHeader>
        <CardTitle>Groups with nothing to measure</CardTitle>
        <CardDescription>
          These groups are in the Dataset Version but not in what could be measured. They are
          listed rather than dropped, because a report that omits a group is easy to misread.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ul className="flex flex-col gap-1 text-sm">
          {report.unmeasured_groups.map((group) => (
            <li key={group.group} data-testid={`unmeasured-${group.group}`}>
              <span className="font-medium">{group.group}</span>
              <span className="text-muted-foreground">{` — ${group.reason}`}</span>
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
  )
}

/** What the report wants you to know before you quote the number. */
function Notes({ report }: { report: FairnessReport }) {
  if (report.notes.length === 0) return null
  return (
    <Card data-testid="fairness-notes">
      <CardHeader>
        <CardTitle>Read this before you quote the number</CardTitle>
      </CardHeader>
      <CardContent>
        <ul className="flex list-disc flex-col gap-1 pl-5 text-sm text-muted-foreground">
          {report.notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
          <li>
            The Model was refitted for this report rather than read back from the run, so these
            numbers describe the same recipe, not the same object.
          </li>
          <li>
            This is one held-out split. A gap this size on a few hundred rows can move a long way
            on a different split.
          </li>
        </ul>
      </CardContent>
    </Card>
  )
}
