/**
 * The Fidelity Report: how close the synthetic rows are to the real ones.
 *
 * Charts are hand-rolled SVG rather than a charting library, for two reasons
 * that matter here: the bars are shared-bin histograms that must line up
 * *exactly* between the two series (a library's auto-scaling would quietly
 * break that), and a plain `<svg>` is real DOM a test can assert on.
 *
 * Every statistic arrives with a sentence, because "p = 0.02" is not something
 * most people can act on.
 */

import { useQuery } from '@tanstack/react-query'
import { motion, useReducedMotion } from 'motion/react'

import { apiGet } from '@/api/client'

// -- types (mirror `api/fidelity.py`) ----------------------------------------

export interface TestResult {
  statistic: number
  p_value: number
}

export interface FidelityReport {
  ks_tests: Record<string, TestResult>
  chi2_tests: Record<string, TestResult>
  correlation_drift: number
  exact_duplicates: { count: number; fraction: number }
  near_copies: { count: number; threshold: number | null; examples: number[]; scanned: number }
  dropped_rows: number
  dropped_ratio: number
  balance: Record<string, Record<string, { requested: number; actual: number }>>
  provider_derived_profile: boolean
  generated_rows: number
  warnings: string[]
}

export interface HistogramSeries {
  column: string
  kind: string
  chart: 'histogram' | 'bar'
  edges?: number[]
  sample?: number[]
  generated?: number[]
  categories?: string[]
  only_in: 'sample' | 'generated' | null
}

export interface CorrelationData {
  columns: string[]
  sample: Array<Array<number | null>>
  generated: Array<Array<number | null>>
  drift: Array<Array<number | null>>
  max_drift: number
}

export interface FidelityResponse {
  version_id: string
  sample_version_id: string | null
  report: FidelityReport
  distributions: HistogramSeries[]
  correlation: CorrelationData
}

export function useFidelity(versionId: string | undefined) {
  return useQuery({
    queryKey: ['fidelity', versionId],
    queryFn: () => apiGet<FidelityResponse>(`/dataset-versions/${versionId}/fidelity`),
    enabled: Boolean(versionId),
    retry: false,
  })
}

// -- plain-language interpretations ------------------------------------------

/**
 * What a test result means. A p-value is only a question ("could this be
 * chance?"), so the wording stays a question too — saying "proves a match"
 * would be a lie, and saying nothing would leave the number unactionable.
 */
export function interpretTest(result: TestResult): { verdict: string; detail: string } {
  const p = result.p_value
  if (p >= 0.05) {
    return {
      verdict: 'looks the same',
      detail: `A difference this size happens by chance about ${Math.round(p * 100)}% of the time, so this column is a reasonable match.`,
    }
  }
  if (p >= 0.01) {
    return {
      verdict: 'probably drifted',
      detail: `Only a ${p.toFixed(3)} chance of seeing a difference this large by accident — worth a look before you trust this column.`,
    }
  }
  return {
    verdict: 'clearly different',
    detail: `A p-value of ${p.toFixed(4)} is not plausibly chance. The real and synthetic ${'distributions'} genuinely differ.`,
  }
}

/** Correlation of −1…1, described in words rather than as a coefficient. */
export function interpretCorrelation(value: number | null): string {
  if (value === null) return 'not enough data'
  const magnitude = Math.abs(value)
  const direction = value < 0 ? 'down' : 'up'
  if (magnitude < 0.1) return 'essentially unrelated'
  if (magnitude < 0.4) return `weakly related ${direction}`
  if (magnitude < 0.7) return `moderately related ${direction}`
  return `strongly related ${direction}`
}

// -- chart geometry -----------------------------------------------------------

const W = 320
const H = 120
const PAD = 4

function useChart() {
  // `useReducedMotion` is null until the media query resolves; treat that as
  // "no preference stated" rather than as an animation to skip.
  const reduced = useReducedMotion() ?? false
  return { reduced, W, H, PAD }
}

function animateIn(reduced: boolean) {
  return {
    initial: reduced ? false : { opacity: 0 },
    animate: { opacity: 1 },
    transition: { duration: reduced ? 0 : 0.3 },
  }
}

function fmt(value: number): string {
  if (Math.abs(value) >= 1000) return value.toLocaleString()
  if (Number.isInteger(value)) return String(value)
  return value.toFixed(2)
}

// -- histogram (numeric + datetime) -------------------------------------------

function HistogramChart({ series }: { series: HistogramSeries }) {
  const { reduced, W: w, H: h } = useChart()
  const edges = series.edges ?? []
  const sample = series.sample ?? []
  const generated = series.generated ?? []
  if (edges.length < 2) {
    return <p className="text-xs text-muted-foreground">Nothing to plot — every value is missing.</p>
  }
  const peak = Math.max(1, ...sample, ...generated)
  const bins = sample.length
  const slot = (w - PAD * 2) / bins
  // half-width so the two series sit side by side within a shared bin
  const barW = Math.max(1, slot / 2 - 0.5)
  const y = (count: number) => h - PAD - (count / peak) * (h - PAD * 2)

  return (
    <div className="flex flex-col gap-1">
      <svg
        viewBox={`0 0 ${w} ${h}`}
        className="w-full"
        role="img"
        aria-label={`Distribution of ${series.column}: real versus synthetic`}
        preserveAspectRatio="none"
      >
        <g {...animateIn(reduced)}>
          {sample.map((count, i) => (
            <rect
              key={`s${i}`}
              x={PAD + i * slot}
              y={y(count)}
              width={barW}
              height={h - PAD - y(count)}
              className="fill-muted-foreground/50"
            />
          ))}
          {generated.map((count, i) => (
            <rect
              key={`g${i}`}
              x={PAD + i * slot + barW}
              y={y(count)}
              width={barW}
              height={h - PAD - y(count)}
              className="fill-primary/70"
            />
          ))}
        </g>
      </svg>
      <div className="flex justify-between text-[10px] text-muted-foreground">
        <span>{fmt(edges[0])}</span>
        <span>peak {fmt(peak)} per bin</span>
        <span>{fmt(edges[edges.length - 1])}</span>
      </div>
    </div>
  )
}

// -- bar (categorical + bool + text) ------------------------------------------

function BarChart({ series }: { series: HistogramSeries }) {
  const { reduced } = useChart()
  const categories = series.categories ?? []
  const sample = series.sample ?? []
  const generated = series.generated ?? []
  if (categories.length === 0) {
    return <p className="text-xs text-muted-foreground">Nothing to plot — every value is missing.</p>
  }

  return (
    <motion.ul
      className="flex flex-col gap-1"
      role="img"
      aria-label={`Categories of ${series.column}: real versus synthetic`}
      {...animateIn(reduced)}
    >
      {categories.map((category, i) => (
        <li key={category} className="flex items-center gap-2 text-xs">
          <span className="w-24 shrink-0 truncate text-muted-foreground" title={category}>
            {category}
          </span>
          <span className="flex h-3.5 flex-1 flex-col justify-center gap-px">
            <span
              className="h-1.5 rounded-sm bg-muted-foreground/50"
              style={{ width: `${Math.max(1, sample[i] * 100)}%` }}
              data-testid={`bar-sample-${series.column}-${i}`}
            />
            <span
              className="h-1.5 rounded-sm bg-primary/70"
              style={{ width: `${Math.max(1, generated[i] * 100)}%` }}
              data-testid={`bar-generated-${series.column}-${i}`}
            />
          </span>
          <span className="w-11 shrink-0 text-right tabular-nums text-muted-foreground">
            {Math.round((sample[i] ?? 0) * 100)}/{Math.round((generated[i] ?? 0) * 100)}%
          </span>
        </li>
      ))}
      <li className="text-[10px] text-muted-foreground">real / synthetic share</li>
    </motion.ul>
  )
}

export function DistributionChart({ series }: { series: HistogramSeries }) {
  return (
    <figure className="flex flex-col gap-1.5">
      <figcaption className="flex flex-wrap items-center gap-2 text-sm">
        <span className="font-medium">{series.column}</span>
        <span className="text-xs text-muted-foreground">{series.kind}</span>
        {series.only_in && (
          <span className="rounded border border-amber-400/40 px-1.5 py-0.5 text-xs text-amber-400">
            only in {series.only_in === 'sample' ? 'the real data' : 'the synthetic rows'}
          </span>
        )}
      </figcaption>
      {series.chart === 'histogram' ? <HistogramChart series={series} /> : <BarChart series={series} />}
    </figure>
  )
}

// -- the correlation drift heatmap --------------------------------------------

/** Drift 0…1 mapped to a readable colour ramp. Null (undefined) stays neutral. */
export function driftColor(value: number | null): string {
  if (value === null) return 'rgb(120 120 120 / 0.15)'
  if (value < 0.1) return 'rgb(74 222 128 / 0.18)'
  if (value < 0.3) return 'rgb(250 204 21 / 0.35)'
  return 'rgb(248 113 113 / 0.55)'
}

export function CorrelationHeatmap({ correlation }: { correlation: CorrelationData }) {
  const { reduced } = useChart()
  const { columns, drift } = correlation
  if (columns.length < 2) {
    return (
      <p className="text-sm text-muted-foreground">
        Correlation drift needs at least two numeric columns to compare.
      </p>
    )
  }
  const cell = 26
  return (
    <div className="flex flex-col gap-2">
      <svg
        viewBox={`0 0 ${columns.length * cell} ${columns.length * cell}`}
        className="w-fit"
        role="img"
        aria-label="Correlation drift between pairs of numeric columns"
      >
        <g {...animateIn(reduced)}>
          {drift.flatMap((row, i) =>
            row.map((value, j) => (
              <rect
                key={`${i}-${j}`}
                x={j * cell}
                y={i * cell}
                width={cell - 2}
                height={cell - 2}
                rx={3}
                fill={driftColor(value)}
                data-testid={`drift-${columns[i]}-${columns[j]}`}
              />
            )),
          )}
        </g>
      </svg>
      <div className="flex flex-wrap gap-3 text-xs text-muted-foreground">
        {columns.map((name) => (
          <span key={name} className="font-mono">
            {name}
          </span>
        ))}
      </div>
      <p className="text-xs text-muted-foreground">
        Worst pair moved by <span className="font-medium text-foreground">{correlation.max_drift.toFixed(3)}</span>{' '}
        in correlation. A pair that is strongly related in the real data but not in the synthetic rows is
        a model that learned the wrong thing.
      </p>
    </div>
  )
}
