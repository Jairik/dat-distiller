/**
 * The Fidelity Report panel: how close the synthetic rows are to the real ones.
 *
 * Shown at the end of the Generate step and reachable from any generated
 * Dataset Version. The Provider-derived banner is deliberately impossible to
 * miss — if the Profile came from the Provider rather than from real data,
 * every comparison below it is weaker than it looks, and the user has to know
 * that before they read the charts.
 */

import { motion, useReducedMotion } from 'motion/react'
import { TriangleAlertIcon } from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import {
  CorrelationHeatmap,
  DistributionChart,
  interpretCorrelation,
  interpretTest,
  useFidelity,
  type FidelityResponse,
  type TestResult,
} from '@/components/fidelity/charts'
import { fmtNumber } from '@/lib/format'

export function FidelityPanel({ versionId }: { versionId: string | undefined }) {
  const query = useFidelity(versionId)
  if (query.isPending) {
    return <p className="text-sm text-muted-foreground">Reading the Fidelity Report…</p>
  }
  if (query.isError) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Fidelity Report</CardTitle>
          <CardDescription>
            {query.error && (query.error as Error & { status?: number }).status === 404
              ? 'This Dataset Version was not produced by Generation, so it has no Fidelity Report.'
              : 'The Fidelity Report could not be read.'}
          </CardDescription>
        </CardHeader>
      </Card>
    )
  }
  return <FidelityBody data={query.data!} />
}

export function FidelityBody({ data }: { data: FidelityResponse }) {
  const { report } = data
  const ksTests = Object.entries(report.ks_tests)
  const chi2Tests = Object.entries(report.chi2_tests)

  return (
    <div className="flex flex-col gap-4">
      {report.provider_derived_profile && <ProviderDerivedBanner />}

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="Rows compared"
          value={fmtNumber(report.generated_rows)}
          hint={
            data.sample_version_id
              ? 'against the sample Dataset Version'
              : 'no sample to compare against — spec-only Generation'
          }
        />
        <StatCard
          label="Worst correlation drift"
          value={data.correlation.max_drift.toFixed(3)}
          hint={
            data.correlation.max_drift < 0.1
              ? 'the relationships hold up'
              : 'at least one relationship moved'
          }
        />
        <StatCard
          label="Near-copies"
          value={fmtNumber(report.near_copies.count)}
          hint={
            report.near_copies.count
              ? 'rows too close to a real row to be safe'
              : 'nothing looks copied from real data'
          }
          tone={report.near_copies.count ? 'warning' : 'ok'}
        />
        <StatCard
          label="Dropped rows"
          value={fmtNumber(report.dropped_rows)}
          hint={`${Math.round(report.dropped_ratio * 100)}% of what was asked for`}
          tone={report.dropped_ratio > 0.1 ? 'warning' : 'ok'}
        />
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Test results</CardTitle>
          <CardDescription>
            Each test asks whether the real and synthetic values could be drawn from the same
            distribution. A low p-value means they could not.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          {ksTests.length === 0 && chi2Tests.length === 0 && (
            <p className="text-sm text-muted-foreground">
              No column had enough values to test.
            </p>
          )}
          {ksTests.map(([column, result]) => (
            <TestRow key={`ks-${column}`} column={column} test="shape" result={result} />
          ))}
          {chi2Tests.map(([column, result]) => (
            <TestRow key={`chi-${column}`} column={column} test="categories" result={result} />
          ))}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Distributions, column by column</CardTitle>
          <CardDescription>
            Real data in grey, synthetic rows in blue, on the same axis so the shapes are directly
            comparable.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-6 md:grid-cols-2">
            {data.distributions.map((series) => (
              <DistributionChart key={series.column} series={series} />
            ))}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Where the relationships moved</CardTitle>
          <CardDescription>
            Pair by pair: where two columns moved together in the real data, do they still move
            together in the synthetic rows?
          </CardDescription>
        </CardHeader>
        <CardContent>
          <CorrelationHeatmap correlation={data.correlation} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Duplicates, balance and the rest</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-3 text-sm">
          <p>
            <span className="font-medium">{fmtNumber(report.exact_duplicates.count)}</span> exact
            duplicate row(s) in the synthetic set (
            {Math.round(report.exact_duplicates.fraction * 100)}%).
          </p>
          {Object.entries(report.balance).map(([column, categories]) => (
            <div key={column} className="flex flex-col gap-1">
              <span className="font-medium">Balance on {column}</span>
              {Object.entries(categories).map(([category, pair]) => (
                <p key={category} className="text-muted-foreground">
                  {category}: asked for {Math.round(pair.requested * 100)}%, got{' '}
                  {Math.round(pair.actual * 100)}%
                  {Math.abs(pair.requested - pair.actual) > 0.05 && (
                    <span className="ml-1 text-amber-400">— off target</span>
                  )}
                </p>
              ))}
            </div>
          ))}
          {report.warnings.length > 0 && (
            <div className="flex flex-col gap-1">
              <span className="font-medium">What this report is flagging</span>
              <ul className="list-inside list-disc text-muted-foreground">
                {report.warnings.map((code) => (
                  <li key={code}>{WARNING_COPY[code] ?? code}</li>
                ))}
              </ul>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

const WARNING_COPY: Record<string, string> = {
  near_copies: 'Some synthetic rows sit too close to real rows — treat this Version as sensitive.',
  dropped_rows_high: 'A lot of Provider rows had to be discarded as invalid.',
  provider_derived_profile: 'The Profile came from the Provider, not from real data.',
  fidelity_drift: 'At least one column does not match the real distribution well.',
}

function ProviderDerivedBanner() {
  const reduced = useReducedMotion()
  return (
    <motion.div
      role="alert"
      initial={reduced ? false : { opacity: 0, y: -4 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: reduced ? 0 : 0.25 }}
      className="flex items-start gap-3 rounded-lg border border-amber-400/40 bg-amber-400/5 p-4"
    >
      <TriangleAlertIcon aria-hidden className="mt-0.5 size-5 shrink-0 text-amber-400" />
      <div className="flex flex-col gap-1">
        <p className="font-medium text-amber-400">The Profile was Provider-derived</p>
        <p className="text-sm text-muted-foreground">
          There was no real data to fit, so the Provider described the shape of the dataset itself.
          Everything below compares the synthetic rows against <em>that description</em>, not against
          reality — treat the charts as a smoke test, not as evidence of realism.
        </p>
      </div>
    </motion.div>
  )
}

function StatCard({
  label,
  value,
  hint,
  tone = 'neutral',
}: {
  label: string
  value: string
  hint: string
  tone?: 'neutral' | 'ok' | 'warning'
}) {
  return (
    <Card>
      <CardHeader className="gap-1">
        <CardDescription className="text-xs">{label}</CardDescription>
        <CardTitle
          className={`text-2xl tabular-nums ${
            tone === 'warning' ? 'text-amber-400' : tone === 'ok' ? 'text-emerald-400' : ''
          }`}
        >
          {value}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-xs text-muted-foreground">{hint}</p>
      </CardContent>
    </Card>
  )
}

function TestRow({
  column,
  test,
  result,
}: {
  column: string
  test: 'shape' | 'categories'
  result: TestResult
}) {
  const { verdict, detail } = interpretTest(result)
  const drifting = result.p_value < 0.05
  return (
    <div className="flex flex-wrap items-start justify-between gap-2 rounded-lg border border-border p-3">
      <div className="flex min-w-0 flex-col gap-0.5">
        <span className="flex items-center gap-2 text-sm font-medium">
          {column}
          <span className="text-xs font-normal text-muted-foreground">
            {test === 'shape' ? 'shape test' : 'category test'}
          </span>
        </span>
        <span className="text-sm text-muted-foreground">{detail}</span>
      </div>
      <div className="flex shrink-0 flex-col items-end gap-1">
        <Badge
          variant="outline"
          className={drifting ? 'border-amber-400/40 text-amber-400' : 'border-emerald-400/40 text-emerald-400'}
        >
          {verdict}
        </Badge>
        <span className="text-xs tabular-nums text-muted-foreground">
          p = {result.p_value.toFixed(4)}
        </span>
      </div>
    </div>
  )
}

export { interpretCorrelation }
