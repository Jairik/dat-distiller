/**
 * ChecksPanel: the closing panel of a step — the responsible-use Checks that
 * came out of it, and the gate that waits for an Acknowledgement.
 *
 * Checks never block a step from being useful, so the copy stays calm and
 * non-hostile; but a warning Check asks for an explicit Acknowledgement before
 * Continue opens. Acknowledged warnings stay visible (marked, timestamped,
 * with their note) because they belong in the Card later.
 *
 * Reused by every step: pass the subject it was evaluated on (a Project, a
 * Dataset Version or a Training Run).
 */

import { useState } from 'react'
import { ChevronDownIcon, InfoIcon, ShieldCheckIcon, TriangleAlertIcon } from 'lucide-react'
import { motion, useReducedMotion } from 'motion/react'

import { useAcknowledgeCheck, useChecks, useUnacknowledgedWarningCount, type Check } from '@/lib/checks'
import { timeAgo } from '@/lib/format'
import { cn } from '@/lib/utils'
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
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'

const SEVERITY = {
  info: {
    label: 'Info',
    icon: InfoIcon,
    iconClass: 'text-muted-foreground',
    badgeClass: 'border-border text-muted-foreground',
    itemClass: 'border-border bg-muted/30',
  },
  warning: {
    label: 'Warning',
    icon: TriangleAlertIcon,
    iconClass: 'text-amber-400',
    badgeClass: 'border-amber-400/40 text-amber-400',
    itemClass: 'border-amber-400/30 bg-amber-400/5',
  },
} as const

function severityOf(check: Check) {
  return SEVERITY[check.severity] ?? SEVERITY.info
}

/** `pii_found` reads better as `pii found` in the small kind hint. */
function kindLabel(kind: string): string {
  return kind.replaceAll('_', ' ')
}

function hasDetails(check: Check): boolean {
  return Object.keys(check.details ?? {}).length > 0
}

/** One Check: message, severity, expandable details, Acknowledgement control. */
function CheckRow({ check }: { check: Check }) {
  const reduced = useReducedMotion()
  const [open, setOpen] = useState(false)
  const [writingNote, setWritingNote] = useState(false)
  const [note, setNote] = useState('')
  const acknowledge = useAcknowledgeCheck()

  const tone = severityOf(check)
  const Icon = tone.icon
  const warning = check.severity === 'warning'
  const detailsId = `check-details-${check.id}`
  const noteId = `check-note-${check.id}`

  const closeNote = () => {
    setWritingNote(false)
    setNote('')
  }

  return (
    <li className={cn('rounded-lg border px-3 py-2.5', tone.itemClass)}>
      <div className="flex items-start gap-2.5">
        <Icon aria-hidden className={cn('mt-0.5 size-4 shrink-0', tone.iconClass)} />
        <div className="min-w-0 flex-1">
          <button
            type="button"
            aria-expanded={open}
            aria-controls={detailsId}
            onClick={() => setOpen((value) => !value)}
            className={cn('flex w-full items-start gap-2 text-left text-sm', !hasDetails(check) && 'cursor-default')}
          >
            <span className="min-w-0 flex-1">
              <span className="font-medium">{check.message}</span>
              <span className="ml-2 text-xs text-muted-foreground">{kindLabel(check.kind)}</span>
            </span>
            {hasDetails(check) && (
              <ChevronDownIcon
                aria-hidden
                className={cn(
                  'mt-0.5 size-4 shrink-0 text-muted-foreground transition-transform',
                  open && 'rotate-180',
                )}
              />
            )}
          </button>

          <div className="mt-1.5 flex flex-wrap items-center gap-2">
            <Badge variant="outline" className={tone.badgeClass}>
              {tone.label}
            </Badge>
            {check.acknowledged && (
              <Badge variant="outline" className="border-emerald-400/40 text-emerald-400">
                Acknowledged
                <span className="font-normal opacity-80">{timeAgo(check.acknowledged_at)}</span>
              </Badge>
            )}
          </div>

          {check.acknowledged && check.note && (
            <p className="mt-1.5 text-xs text-muted-foreground italic">“{check.note}”</p>
          )}

          {open && hasDetails(check) && (
            <motion.div
              id={detailsId}
              initial={reduced ? false : { height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              transition={{ duration: reduced ? 0 : 0.22, ease: 'easeOut' }}
              style={{ overflow: 'hidden' }}
              className="mt-2"
            >
              <p className="text-xs font-medium tracking-wide text-muted-foreground uppercase">
                Details
              </p>
              <pre className="mt-1 overflow-x-auto rounded-md border border-border bg-background/60 p-2 text-xs leading-relaxed text-muted-foreground">
                {JSON.stringify(check.details ?? {}, null, 2)}
              </pre>
            </motion.div>
          )}

          {warning && !check.acknowledged && (
            <div className="mt-2.5">
              {writingNote ? (
                <div className="flex flex-col gap-2">
                  <Label htmlFor={noteId} className="text-xs font-normal text-muted-foreground">
                    Add a note about why you are continuing (optional)
                  </Label>
                  <Textarea
                    id={noteId}
                    value={note}
                    rows={2}
                    className="min-h-14 text-sm"
                    placeholder="For example: these addresses are synthetic, so they are not real PII."
                    onChange={(event) => setNote(event.target.value)}
                  />
                  <div className="flex flex-wrap items-center gap-2">
                    <Button
                      size="sm"
                      onClick={() => acknowledge.mutate({ checkId: check.id, note })}
                      disabled={acknowledge.isPending}
                    >
                      {acknowledge.isPending ? 'Recording…' : 'Record Acknowledgement'}
                    </Button>
                    <Button size="sm" variant="ghost" onClick={closeNote}>
                      Cancel
                    </Button>
                  </div>
                </div>
              ) : (
                <Button size="sm" variant="outline" onClick={() => setWritingNote(true)}>
                  Acknowledge
                </Button>
              )}
              {acknowledge.isError && (
                <p role="alert" className="mt-1.5 text-sm text-destructive">
                  That Acknowledgement did not stick — try once more.
                </p>
              )}
            </div>
          )}
        </div>
      </div>
    </li>
  )
}

export interface ChecksPanelProps {
  /** What the Checks were evaluated on: `project` · `dataset_version` · `training_run`. */
  subjectType: string
  subjectId: string | undefined
  /** Label of the gate button; it stays disabled while warnings are open. */
  continueLabel?: string
  onContinue?: () => void
}

export function ChecksPanel({
  subjectType,
  subjectId,
  continueLabel = 'Continue',
  onContinue,
}: ChecksPanelProps) {
  const query = useChecks(subjectType, subjectId)
  const checks = query.data?.checks ?? []
  const outstanding = query.data?.unacknowledged_warnings ?? 0
  // Loading: keep the gate shut for a moment. Failed: stay out of the user's
  // way — a Check never blocks the work itself.
  const ready = !query.isPending
  const clear = query.isError || outstanding === 0

  return (
    <Card className="gap-4">
      <CardHeader>
        <CardTitle>Before you continue</CardTitle>
        <CardDescription>
          These notes came from Dat Distiller&apos;s responsible-use checks. Nothing here is
          blocking your data — warnings just want a quick look, and an Acknowledgement if you
          decide to move on.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        {query.isPending && (
          <p className="text-sm text-muted-foreground">Reviewing this step&hellip;</p>
        )}
        {query.isError && (
          <p className="text-sm text-muted-foreground">
            Checks could not be read for this step, so there is nothing to acknowledge.
          </p>
        )}
        {ready && !query.isError && checks.length === 0 && (
          <p className="flex items-center gap-2 text-sm text-muted-foreground">
            <ShieldCheckIcon aria-hidden className="size-4 text-emerald-400" />
            No checks — looking good.
          </p>
        )}
        {checks.length > 0 && (
          <>
            {outstanding > 0 && (
              <p role="status" className="text-sm text-muted-foreground">
                <span className="font-medium text-amber-400">
                  {outstanding === 1 ? '1 warning' : `${outstanding} warnings`}
                </span>{' '}
                {outstanding === 1 ? 'needs' : 'need'} your Acknowledgement before you can continue.
              </p>
            )}
            <ul className="flex flex-col gap-2">
              {checks.map((check) => (
                <CheckRow key={check.id} check={check} />
              ))}
            </ul>
          </>
        )}
      </CardContent>
      {onContinue && (
        <CardFooter className="flex-wrap items-center justify-between gap-3">
          <p className="text-sm text-muted-foreground">
            {/* Opening the gate on a failed read is deliberate — a Check never
                blocks the work — but "Everything here has been read" is not, when
                nothing was read. Note the e2e helper waits for exactly this
                string, so it could pass on an errored Checks query. */}
            {query.isError
              ? 'The Checks could not be read, so this step is open on trust.'
              : ready && clear
                ? 'Everything here has been read.'
                : 'Acknowledge the warnings above to unlock this step.'}
          </p>
          <Button onClick={onContinue} disabled={!ready || !clear}>
            {continueLabel}
          </Button>
        </CardFooter>
      )}
    </Card>
  )
}

/**
 * Compact header marker for a subject with open warning Checks — quiet at zero,
 * so pages can drop it in without adding noise.
 */
export function ChecksWarningBadge({
  subjectType,
  subjectId,
  className,
}: {
  subjectType: string
  subjectId: string | undefined
  className?: string
}) {
  const count = useUnacknowledgedWarningCount(subjectType, subjectId)
  if (count <= 0) return null
  return (
    <Badge variant="outline" className={cn('border-amber-400/40 text-amber-400', className)}>
      <TriangleAlertIcon aria-hidden />
      {count === 1 ? '1 warning' : `${count} warnings`} to acknowledge
    </Badge>
  )
}
