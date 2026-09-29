/**
 * PII findings on a Dataset Version, and what you can do about them.
 *
 * The panel extends the Checks panel rather than replacing it: the `pii_found`
 * warning Check is what "Continue" acts on, so acknowledging it *is* the warn
 * path (issue #40) and this UI just gives that a button and a reason.
 *
 * **Nothing here ever shows raw PII.** Examples arrive masked from the backend
 * and are rendered exactly as sent. A finding with no examples renders nothing
 * in their place — the UI never falls back to fetching the real value, because
 * that fallback is precisely how PII would end up on someone's screen.
 */

import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, useReducedMotion } from 'motion/react'
import { EyeIcon, ShieldCheckIcon, TriangleAlertIcon } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import {
  PII_ACTIONS,
  examplesOf,
  useApplyPiiActions,
  usePiiScan,
  type PiiAction,
  type PiiFinding,
} from '@/lib/pii'
import { useAcknowledgeCheck, useChecks } from '@/lib/checks'
import { useProjectContext } from '@/routes/project-page'

export function PiiPanel({ versionId }: { versionId: string | undefined }) {
  const { project, reloadVersions } = useProjectContext()
  const navigate = useNavigate()
  const scan = usePiiScan(versionId)
  const apply = useApplyPiiActions(versionId)
  const [choice, setChoice] = useState<Record<string, PiiAction>>({})
  const [done, setDone] = useState<string | null>(null)

  const columns = useMemo(() => {
    const byColumn = new Map<string, PiiFinding[]>()
    for (const finding of scan.data?.findings ?? []) {
      byColumn.set(finding.column, [...(byColumn.get(finding.column) ?? []), finding])
    }
    return [...byColumn.entries()]
  }, [scan.data])

  const chosen = Object.entries(choice)
  const canApply = chosen.length > 0 && !apply.isPending

  if (scan.isPending) {
    return <p className="text-sm text-muted-foreground">Scanning for PII…</p>
  }
  if (scan.isError) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>PII</CardTitle>
          <CardDescription>The scan could not be read for this Dataset Version.</CardDescription>
        </CardHeader>
      </Card>
    )
  }
  if (!scan.data || columns.length === 0) {
    return null // nothing to say: the Checks panel already reports a clean scan
  }

  return (
    <div className="flex flex-col gap-4">
      <Card>
        <CardHeader>
          <CardTitle>PII findings</CardTitle>
          <CardDescription>
            {scan.data.summary.total_findings} possible match(es) across{' '}
            {columns.length} column(s). Examples are already masked — the real values are never
            sent to this page.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          {columns.map(([column, findings]) => (
            <FindingRow
              key={column}
              column={column}
              findings={findings}
              choice={choice[column]}
              onChoose={(action) =>
                setChoice((current) => ({ ...current, [column]: action }))
              }
            />
          ))}

          <div className="flex flex-wrap items-center gap-3">
            <Button
              size="sm"
              disabled={!canApply}
              onClick={() =>
                apply.mutate(choice, {
                  onSuccess: (result) => {
                    setChoice({})
                    setDone(JSON.stringify(result.summary))
                    reloadVersions()
                  },
                })
              }
            >
              {apply.isPending
                ? 'Applying…'
                : `Apply to ${chosen.length} column${chosen.length === 1 ? '' : 's'}`}
            </Button>
            <span className="text-xs text-muted-foreground">
              {chosen.length === 0
                ? 'Pick Mask or Drop on a column to act on it.'
                : `${chosen.map(([c, a]) => `${c}: ${a}`).join(', ')}`}
            </span>
          </div>

          {done && (
            <p role="status" className="text-sm text-emerald-400">
              {`Done. A new Dataset Version was created — ${done}. It is now in the history.`}
            </p>
          )}
          {apply.isError && (
            <p role="alert" className="text-sm text-destructive">
              {(apply.error as Error).message}
            </p>
          )}
        </CardContent>
      </Card>

      <ContinueCard versionId={versionId} projectId={project.id} navigate={navigate} />
    </div>
  )
}

function FindingRow({
  column,
  findings,
  choice,
  onChoose,
}: {
  column: string
  findings: PiiFinding[]
  choice: PiiAction | undefined
  onChoose: (action: PiiAction) => void
}) {
  const reduced = useReducedMotion()
  const total = findings.reduce((sum, f) => sum + f.count, 0)
  const detectors = [...new Set(findings.map((f) => f.detector))]

  return (
    <motion.div
      initial={reduced ? false : { opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: reduced ? 0 : 0.2 }}
      className="flex flex-col gap-2 rounded-lg border border-amber-400/30 bg-amber-400/5 p-3"
      data-testid={`pii-${column}`}
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="flex flex-wrap items-center gap-2 text-sm font-medium">
          <TriangleAlertIcon aria-hidden className="size-4 text-amber-400" />
          {column}
          <Badge variant="outline" className="text-xs">
            {detectors.join(', ')}
          </Badge>
          <span className="text-xs tabular-nums text-muted-foreground">{total} match(es)</span>
        </span>
        <span className="flex gap-1.5">
          {PII_ACTIONS.map((action) => (
            <Button
              key={action}
              size="sm"
              variant={choice === action ? 'default' : 'outline'}
              // The variant is a colour and a border, and nothing else: without
              // this a screen-reader user activating "Mask" gets no confirmation
              // that anything was selected.
              aria-pressed={choice === action}
              onClick={() => onChoose(action)}
            >
              {action === 'mask' ? 'Mask' : 'Drop'}
            </Button>
          ))}
        </span>
      </div>

      <div className="flex flex-col gap-1">
        <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <EyeIcon aria-hidden className="size-3" />
          Masked examples
        </span>
        {examplesOf(findings[0]).length === 0 ? (
          <p className="text-xs text-muted-foreground">
            No examples were kept for this column — the real values are not shown anywhere.
          </p>
        ) : (
          <ul className="flex flex-wrap gap-1.5">
            {findings.flatMap((f) => examplesOf(f)).slice(0, 6).map((example, i) => (
              <li
                key={i}
                className="rounded border border-border bg-background/60 px-1.5 py-0.5 font-mono text-xs"
              >
                {example}
              </li>
            ))}
          </ul>
        )}
      </div>

      {choice && (
        <p className="text-xs text-muted-foreground">
          {choice === 'mask'
            ? 'Masking rewrites the detected spans as [REDACTED] in a new Dataset Version.'
            : 'Dropping removes this whole column in a new Dataset Version.'}
        </p>
      )}
    </motion.div>
  )
}

/**
 * The `warn` path. There is deliberately no "warn" action that touches data —
 * acknowledging the `pii_found` Check *is* the warn path, and that is the only
 * thing Continue does. The Check is found by querying this version's Checks
 * rather than being passed down, so the panel works wherever it is dropped in.
 */
function ContinueCard({
  versionId,
  projectId,
  navigate,
}: {
  versionId: string | undefined
  projectId: string
  navigate: ReturnType<typeof useNavigate>
}) {
  const checks = useChecks('dataset_version', versionId)
  const acknowledge = useAcknowledgeCheck()
  const [acknowledged, setAcknowledged] = useState(false)

  const piiChecks = (checks.data?.checks ?? []).filter((c) => c.kind === 'pii_found')
  const outstanding = checks.data?.unacknowledged_warnings ?? 0
  const next = piiChecks.find((c) => !c.acknowledged)

  return (
    <Card>
      <CardHeader>
        <CardTitle>Or carry on as they are</CardTitle>
        <CardDescription>
          If the matches are synthetic — generated by a Provider, or already made up — you can
          acknowledge them and keep this Dataset Version as it is. No new version is created.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <Button
          size="sm"
          variant="outline"
          disabled={acknowledge.isPending || acknowledged || outstanding === 0 || !next}
          onClick={() =>
            next &&
            acknowledge.mutate(
              { checkId: next.id, note: 'matches are synthetic, continuing' },
              { onSuccess: () => setAcknowledged(true) },
            )
          }
        >
          <ShieldCheckIcon aria-hidden />
          {acknowledged ? 'Acknowledged' : 'Continue as they are'}
        </Button>
        {checks.data && outstanding === 0 && !acknowledged && (
          <p className="text-xs text-muted-foreground">
            Every PII warning on this version is already acknowledged.
          </p>
        )}
        {acknowledge.isError && (
          <p role="alert" className="text-sm text-destructive">
            {(acknowledge.error as Error).message}
          </p>
        )}
        {acknowledged && (
          <Button
            size="sm"
            variant="ghost"
            onClick={() => navigate(`/projects/${projectId}/dataset?v=${versionId}`)}
          >
            Stay on this Dataset Version
          </Button>
        )}
      </CardContent>
    </Card>
  )
}
