/**
 * Evaluation plots, drawn with Recharts.
 *
 * A chart is not a table. Every one of these renders **both**: the Recharts
 * drawing for the shape, and the actual numbers beside it. A ROC curve you can
 * only look at is a curve you cannot check, and a feature-importance bar with
 * no labels is a bar chart that says nothing at all.
 *
 * The numbers are the reason this component exists; the SVG is the reason it is
 * not just a table. Nothing here fetches — `usePlot` is called by the caller
 * once the user has asked for a plot.
 */

import { useMemo } from 'react'
import { motion, useReducedMotion } from 'motion/react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { Badge } from '@/components/ui/badge'
import {
  PLOT_HINTS,
  type ConfusionMatrix,
  type FeatureImportance,
  type PlotName,
  type PrecisionRecall,
  type Residuals,
  type RocCurve,
} from '@/lib/evaluate'

/** One fixed size, so the charts do not need a measured parent. */
const CHART = { width: 420, height: 220 }

/** Recharts hands a formatter `ValueType | undefined`. */
const threeDecimals = (value: unknown) => Number(value).toFixed(3)
const fourDecimals = (value: unknown) => Number(value).toFixed(4)

/** Categorical colours, so a stack is readable without a config file. */
const PALETTE = [
  'var(--color-primary)',
  'var(--color-muted-foreground)',
  'oklch(0.7 0.15 70)',
  'var(--color-destructive)',
]

export function PlotCard({
  plot,
  title,
  children,
  empty,
}: {
  plot: PlotName
  title: string
  children?: React.ReactNode
  empty?: string | null
}) {
  const reduced = useReducedMotion()
  return (
    <motion.div
      initial={reduced ? false : { opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: reduced ? 0 : 0.24 }}
      data-testid={`plot-${plot}`}
      className="flex flex-col gap-2 rounded-lg border border-border p-3"
    >
      <span className="flex flex-col">
        <span className="text-sm font-medium">{title}</span>
        <span className="text-xs text-muted-foreground">{PLOT_HINTS[plot]}</span>
      </span>
      {empty ? <p className="text-sm text-amber-400">{empty}</p> : children}
    </motion.div>
  )
}

function ChartFrame({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div role="img" aria-label={label} data-testid="chart">
      <ResponsiveContainer width="100%" height={CHART.height}>
        {children as React.ReactElement}
      </ResponsiveContainer>
    </div>
  )
}

// -- confusion matrix ---------------------------------------------------------

export function ConfusionMatrixView({ data }: { data: ConfusionMatrix }) {
  // one bar per actual class, stacked by what was predicted: the shape
  // Recharts can render cleanly for a grid
  const bars = useMemo(
    () =>
      data.matrix.map((row, i) => {
        const entry: Record<string, string | number> = { actual: data.labels[i] }
        data.labels.forEach((label, j) => {
          entry[label] = row[j] ?? 0
        })
        return entry
      }),
    [data],
  )
  // a heat colour, so the diagonal stands out without hiding the numbers
  const shade = (share: number) => {
    const alpha = 0.08 + Math.min(0.72, share * 0.8)
    return `color-mix(in oklab, var(--color-primary) ${Math.round(alpha * 100)}%, transparent)`
  }

  return (
    <>
      <ChartFrame
        label={`Confusion matrix: one bar per actual class, stacked by what was predicted, over ${data.n} held-out rows`}
      >
        <BarChart {...CHART} data={bars} margin={{ top: 12, right: 12, bottom: 12, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
          <XAxis
            dataKey="actual"
            label={{ value: 'actual', position: 'insideBottom', offset: -4, fontSize: 10 }}
          />
          <YAxis
            width={40}
            allowDecimals={false}
            label={{ value: 'rows', angle: -90, position: 'insideLeft', fontSize: 10 }}
          />
          <Tooltip />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {data.labels.map((label, index) => (
            <Bar key={label} dataKey={label} stackId="a" fill={PALETTE[index % PALETTE.length]} />
          ))}
        </BarChart>
      </ChartFrame>

      {/* The numbers are the point: rows are what happened, columns what was predicted. */}
      <table className="w-full text-xs" data-testid="confusion-table">
        <caption className="sr-only">
          Confusion matrix. Rows are the actual class, columns the predicted class.
        </caption>
        <thead>
          <tr>
            <th scope="col" className="px-1 py-0.5 text-left font-medium">
              actual \ predicted
            </th>
            {data.labels.map((label) => (
              <th key={label} scope="col" className="px-1 py-0.5 text-right font-medium">
                {label}
              </th>
            ))}
            <th scope="col" className="px-1 py-0.5 text-right font-medium">
              support
            </th>
          </tr>
        </thead>
        <tbody>
          {data.matrix.map((row, i) => (
            <tr key={data.labels[i]}>
              <th scope="row" className="px-1 py-0.5 text-left font-medium">
                {data.labels[i]}
              </th>
              {row.map((count, j) => (
                <td
                  key={j}
                  style={{ backgroundColor: shade(data.row_normalised[i]?.[j] ?? 0) }}
                  className="px-1 py-0.5 text-right tabular-nums"
                >
                  {count}
                </td>
              ))}
              <td className="px-1 py-0.5 text-right tabular-nums text-muted-foreground">
                {data.support[i]}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-xs text-muted-foreground">{`${data.n} held-out row(s).`}</p>
    </>
  )
}

// -- ROC / PR -----------------------------------------------------------------

export function RocView({ data }: { data: RocCurve }) {
  const series = useMemo(() => {
    if (data.kind === 'binary' && data.fpr && data.tpr) {
      return [
        {
          name: `ROC (${data.positive_class})`,
          points: data.fpr.map((x, i) => ({ x, y: data.tpr?.[i] ?? 0 })),
        },
      ]
    }
    return Object.entries(data.curves ?? {}).map(([label, curve]) => ({
      name: label,
      points: curve.fpr.map((x, i) => ({ x, y: curve.tpr[i] ?? 0 })),
    }))
  }, [data])

  const auc =
    data.kind === 'binary' ? data.auc : data.auc_macro ?? null

  return (
    <>
      <ChartFrame label={`Receiver operating characteristic curve, AUC ${fmtNumber(auc)}`}>
        <LineChart {...CHART} margin={{ top: 12, right: 12, bottom: 12, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
          <XAxis
            dataKey="x"
            type="number"
            domain={[0, 1]}
            tickFormatter={(v) => v.toFixed(1)}
            label={{ value: 'false positive rate', position: 'insideBottom', offset: -4, fontSize: 10 }}
          />
          <YAxis
            dataKey="y"
            type="number"
            domain={[0, 1]}
            width={44}
            label={{ value: 'true positive rate', angle: -90, position: 'insideLeft', fontSize: 10 }}
          />
          <Tooltip formatter={threeDecimals} />
          {/* the no-skill diagonal: a Model at chance sits on this line */}
          <ReferenceLine
            segment={[
              { x: 0, y: 0 },
              { x: 1, y: 1 },
            ]}
            stroke="var(--color-muted-foreground)"
            strokeDasharray="4 4"
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {series.map((s) => (
            <Line key={s.name} data={s.points} dataKey="y" name={s.name} dot={false} strokeWidth={2} />
          ))}
        </LineChart>
      </ChartFrame>
      <p className="text-sm" data-testid="auc">
        {`AUC ${fmtNumber(auc)}`}
        {data.kind === 'binary'
          ? ` — positive class ${data.positive_class}.`
          : ' — macro over the one-vs-rest curves.'}
      </p>
    </>
  )
}

export function PrecisionRecallView({ data }: { data: PrecisionRecall }) {
  const series = useMemo(() => {
    if (data.kind === 'binary' && data.precision && data.recall) {
      return [
        {
          name: `PR (${data.positive_class})`,
          points: data.recall.map((x, i) => ({ x, y: data.precision?.[i] ?? 0 })),
        },
      ]
    }
    return Object.entries(data.curves ?? {}).map(([label, curve]) => ({
      name: label,
      points: curve.recall.map((x, i) => ({ x, y: curve.precision[i] ?? 0 })),
    }))
  }, [data])

  const ap =
    data.kind === 'binary' ? data.average_precision : data.average_precision_macro ?? null

  return (
    <>
      <ChartFrame label={`Precision-recall curve, average precision ${fmtNumber(ap)}`}>
        <LineChart {...CHART} margin={{ top: 12, right: 12, bottom: 12, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
          <XAxis
            dataKey="x"
            type="number"
            domain={[0, 1]}
            tickFormatter={(v) => v.toFixed(1)}
            label={{ value: 'recall', position: 'insideBottom', offset: -4, fontSize: 10 }}
          />
          <YAxis
            dataKey="y"
            type="number"
            domain={[0, 1]}
            width={44}
            label={{ value: 'precision', angle: -90, position: 'insideLeft', fontSize: 10 }}
          />
          <Tooltip formatter={threeDecimals} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {series.map((s) => (
            <Line key={s.name} data={s.points} dataKey="y" name={s.name} dot={false} strokeWidth={2} />
          ))}
        </LineChart>
      </ChartFrame>
      <p className="text-sm" data-testid="average-precision">
        {`Average precision ${fmtNumber(ap)}`}
        {data.kind === 'binary' ? ` — positive class ${data.positive_class}.` : ' — macro.'}
      </p>
    </>
  )
}

// -- residuals ----------------------------------------------------------------

export function ResidualsView({ data }: { data: Residuals }) {
  const points = useMemo(
    () => data.predicted.map((predicted, i) => ({ predicted, residual: data.residuals[i] ?? 0 })),
    [data],
  )
  return (
    <>
      <ChartFrame label={`Residuals against predicted value, mean absolute error ${fmtNumber(data.mae)}`}>
        <ScatterChart {...CHART} margin={{ top: 12, right: 12, bottom: 12, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
          <XAxis
            dataKey="predicted"
            type="number"
            label={{ value: 'predicted', position: 'insideBottom', offset: -4, fontSize: 10 }}
          />
          <YAxis
            dataKey="residual"
            type="number"
            label={{ value: 'actual − predicted', angle: -90, position: 'insideLeft', fontSize: 10 }}
          />
          <Tooltip formatter={threeDecimals} />
          {/* a biased Model's points sit off this line */}
          <ReferenceLine y={0} stroke="var(--color-muted-foreground)" strokeDasharray="4 4" />
          <Scatter data={points} fill="var(--color-primary)" />
        </ScatterChart>
      </ChartFrame>
      <p className="text-sm" data-testid="residual-summary">
        {`MAE ${fmtNumber(data.mae)} · mean residual ${fmtNumber(data.mean_residual)} over ${data.n} held-out row(s).`}
      </p>
    </>
  )
}

// -- feature importance -------------------------------------------------------

export function FeatureImportanceView({ data }: { data: FeatureImportance }) {
  return (
    <>
      <ChartFrame label="Feature importance, by permutation">
        <BarChart
          {...CHART}
          layout="vertical"
          data={data.importance}
          margin={{ top: 8, right: 40, bottom: 8, left: 8 }}
        >
          <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
          <XAxis
            type="number"
            label={{ value: 'drop in score when shuffled', position: 'insideBottom', offset: -4, fontSize: 10 }}
          />
          <YAxis type="category" dataKey="feature" width={96} />
          <Tooltip formatter={fourDecimals} />
          <Bar dataKey="drop" radius={[0, 3, 3, 0]}>
            {data.importance.map((entry, index) => (
              <Cell
                key={entry.feature}
                fill={index === 0 ? 'var(--color-primary)' : 'var(--color-muted-foreground)'}
              />
            ))}
          </Bar>
        </BarChart>
      </ChartFrame>
      <ul className="flex flex-col gap-0.5 text-xs" data-testid="importance-list">
        {data.importance.map((entry) => (
          <li key={entry.feature} className="flex justify-between gap-2">
            <span className="font-mono">{entry.feature}</span>
            <span className="tabular-nums text-muted-foreground">{fmtNumber(entry.drop)}</span>
          </li>
        ))}
      </ul>
      <p className="text-xs text-muted-foreground">
        {`Baseline score ${fmtNumber(data.baseline)}, averaged over ${data.repeats} shuffle(s). A negative drop means shuffling did not hurt.`}
      </p>
    </>
  )
}

function fmtNumber(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return value.toFixed(4)
}

export { Badge }
