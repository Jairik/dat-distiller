/**
 * The Review Queue: Jev's least confident labels, one at a time.
 *
 * Built for speed, because reviewing is the tedium in this app. Everything is
 * reachable from the keyboard — `a` accept, `o` override, `x` exclude, `j`/`k`
 * to move — and the keyboard only works while the reviewer is not typing into
 * a field, so overriding a value never accidentally triggers a shortcut.
 *
 * Decisions are staged locally and applied in one call: a reviewer working
 * down a list should be able to accept a dozen rows and commit once. Nothing
 * is written until Apply, and Apply says plainly what it did.
 */

import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import { CheckIcon, TriangleAlertIcon, XIcon } from 'lucide-react'

import { ChecksPanel } from '@/components/checks/checks-panel'
import { AnimatedNumber } from '@/components/motion'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label as LabelField } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'
import {
  useApplyReview,
  useReviewQueue,
  useReviewStatus,
  type Decision,
  type QueueItem,
} from '@/lib/review'
import { serializeState } from '@/lib/label'
import { useProjectContext } from '@/routes/project-page'
import { useQuery } from '@tanstack/react-query'
import { apiGet } from '@/api/client'

export function ReviewQueuePanel({ versionId }: { versionId: string | undefined }) {
  const { project, reloadVersions } = useProjectContext()
  const navigate = useNavigate()
  const [threshold, setThreshold] = useState(0.8)
  const [staged, setStaged] = useState<Record<string, QueueItem>>({})
  const [cursor, setCursor] = useState(0)
  const [reviewed, setReviewed] = useState<string | null>(null)
  const [newVersionId, setNewVersionId] = useState<string | null>(null)

  const queue = useReviewQueue(versionId, threshold)
  const status = useReviewStatus(versionId, threshold)
  const apply = useApplyReview(versionId)
  const row = useRowPreview(versionId)

  const items = queue.data?.items ?? []
  const families = queue.data?.families ?? {}

  const key = (item: QueueItem) => `${item.row_index}:${item.family}`
  const decisions = useMemo(() => Object.values(staged), [staged])
  const remaining = status.data?.unreviewed_count ?? 0

  const decide = useCallback(
    (item: QueueItem, decision: Decision, override?: unknown) => {
      setStaged((current) => ({
        ...current,
        [key(item)]: { ...item, decision, override: override ?? null },
      }))
    },
    [],
  )

  const move = useCallback(
    (delta: number) => {
      setCursor((c) => Math.min(Math.max(0, c + delta), Math.max(items.length - 1, 0)))
    },
    [items.length],
  )

  // Keyboard review. Ignored while a field has focus, so typing an override
  // value never fires a shortcut.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null
      if (target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return
      const item = items[cursor]
      if (!item) return
      const key_ = event.key.toLowerCase()
      if (key_ === 'j' || event.key === 'ArrowDown') {
        event.preventDefault()
        move(1)
      } else if (key_ === 'k' || event.key === 'ArrowUp') {
        event.preventDefault()
        move(-1)
      } else if (key_ === 'a') {
        event.preventDefault()
        decide(item, 'accept')
        move(1)
      } else if (key_ === 'x') {
        event.preventDefault()
        decide(item, 'exclude')
        move(1)
      } else if (key_ === 'o') {
        // move focus to the override field so the reviewer can type straight away
        event.preventDefault()
        const field = document.getElementById(`override-${item.row_index}-${item.family}`)
        if (field instanceof HTMLElement) field.focus()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [items, cursor, decide, move])

  if (!versionId) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Review Queue</CardTitle>
          <CardDescription>Choose a labeled Dataset Version to review.</CardDescription>
        </CardHeader>
      </Card>
    )
  }

  return (
    <div className="flex flex-col gap-4">
      <Card>
        <CardHeader>
          <CardTitle>Review Queue</CardTitle>
          <CardDescription>
            Labels Jev was unsure about. Accept what it got right, override what it got wrong, or
            exclude the row. Nothing is saved until you apply.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <div className="flex flex-wrap items-end gap-4">
            <div className="flex min-w-56 flex-1 flex-col gap-2">
              <LabelField htmlFor="review-threshold">
                Review threshold
                <span className="ml-2 tabular-nums text-muted-foreground">
                  {(threshold * 100).toFixed(0)}%
                </span>
              </LabelField>
              <Slider
                id="review-threshold"
                min={0}
                max={100}
                step={5}
                value={[threshold * 100]}
                onValueChange={([value]) => setThreshold((value ?? 80) / 100)}
              />
              <p className="text-xs text-muted-foreground">
                Labels Jev answered with less confidence than this are queued. Lowering it widens
                the queue; it does not change what is trained on.
              </p>
            </div>
            <div className="flex flex-col items-end">
              <span className="text-xs text-muted-foreground">Still outstanding</span>
              <AnimatedNumber
                value={remaining}
                className="text-2xl font-semibold tabular-nums"
                format={(v) => String(Math.round(v))}
              />
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <Button
              size="sm"
              variant="outline"
              onClick={() =>
                items.forEach((item) => decide(item, 'accept'))
              }
              disabled={items.length === 0}
            >
              Accept all {items.length} shown
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                setStaged({})
                setCursor(0)
              }}
              disabled={decisions.length === 0}
            >
              Clear staged decisions
            </Button>
            <Button
              size="sm"
              disabled={decisions.length === 0 || apply.isPending}
              onClick={() =>
                apply.mutate(
                  {
                    decisions: decisions.map((d) => ({
                      row_index: d.row_index,
                      family: d.family,
                      question_type: d.question_type,
                      decision: d.decision as Decision,
                      override: d.override,
                    })),
                    threshold,
                  },
                  {
                    onSuccess: (result) => {
                      setReviewed(JSON.stringify(result.outcome))
                      setStaged({})
                      setNewVersionId(result.version.id)
                      reloadVersions()
                    },
                  },
                )
              }
            >
              {apply.isPending
                ? 'Applying…'
                : `Apply ${decisions.length} decision${decisions.length === 1 ? '' : 's'}`}
            </Button>
            <span className="text-xs text-muted-foreground">
              {decisions.length} staged
            </span>
          </div>

          {reviewed && (
            <p role="status" className="text-sm text-emerald-400">
              {`Applied: ${reviewed}. The new Dataset Version is in the history.`}
            </p>
          )}
          {apply.isError && (
            <p role="alert" className="text-sm text-destructive">
              {(apply.error as Error).message}
            </p>
          )}
        </CardContent>
      </Card>

      {(status.data?.unlabeled_count ?? 0) > 0 && (
        <p className="flex items-start gap-2 text-sm text-amber-400">
          <TriangleAlertIcon aria-hidden className="size-4 shrink-0" />
          {`${status.data?.unlabeled_count} label(s) were never answered by Jev. They cannot be ` +
            'reviewed — nothing was ever proposed — and they count as unreviewed until the Train step decides otherwise.'}
        </p>
      )}

      {queue.isPending && <p className="text-sm text-muted-foreground">Reading the queue…</p>}
      {queue.isError && (
        <p role="alert" className="text-sm text-destructive">
          {(queue.error as Error).message}
        </p>
      )}
      {queue.data && items.length === 0 && (
        <Card>
          <CardContent>
            <p className="flex items-center gap-2 text-sm text-muted-foreground">
              <CheckIcon aria-hidden className="size-4 text-emerald-400" />
              Nothing to review at this threshold — every label Jev gave is at least this confident.
            </p>
          </CardContent>
        </Card>
      )}

      {items.length > 0 && (
        <>
          <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
            <span>
              Reviewing {cursor + 1} of {items.length}
            </span>
            <Shortcut keys="J / K" what="move" />
            <Shortcut keys="A" what="accept" />
            <Shortcut keys="O" what="override" />
            <Shortcut keys="X" what="exclude" />
          </div>
          <ul className="flex flex-col gap-2">
            <AnimatePresence initial={false}>
              {items.map((queued, index) => {
                // render the *staged* copy, so a decision shows the moment it is
                // made rather than only after Apply
                const item = staged[key(queued)] ?? queued
                return (
                  <QueueCard
                    key={key(queued)}
                    item={item}
                    state={row.data ? serializeState(row.data, Object.keys(row.data)) : null}
                    active={index === cursor}
                    familyType={families[item.family] ?? item.question_type}
                    onFocus={() => setCursor(index)}
                    onDecide={(decision, override) => {
                      decide(queued, decision, override)
                      if (decision !== 'override') move(1)
                    }}
                  />
                )
              })}
            </AnimatePresence>
          </ul>
        </>
      )}

      {newVersionId && (
        <div className="flex flex-wrap gap-2">
          <Button
            size="sm"
            variant="outline"
            onClick={() => navigate(`/projects/${project.id}/dataset?v=${newVersionId}`)}
          >
            Open the reviewed Dataset Version
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => {
              setNewVersionId(null)
              setReviewed(null)
            }}
          >
            Keep reviewing
          </Button>
        </div>
      )}

      {newVersionId && (
        <ChecksPanel
          subjectType="dataset_version"
          subjectId={newVersionId}
          continueLabel="Continue to Train"
          onContinue={() => navigate(`/projects/${project.id}/train?version=${newVersionId}`)}
        />
      )}
    </div>
  )
}

function Shortcut({ keys, what }: { keys: string; what: string }) {
  return (
    <span className="flex items-center gap-1">
      <kbd className="rounded border border-border bg-muted px-1 py-0.5 font-mono text-[10px]">
        {keys}
      </kbd>
      {what}
    </span>
  )
}

function QueueCard({
  item,
  state,
  active,
  familyType,
  onFocus,
  onDecide,
}: {
  item: QueueItem
  state: string | null
  active: boolean
  familyType: string
  onFocus: () => void
  onDecide: (decision: Decision, override?: unknown) => void
}) {
  const reduced = useReducedMotion()
  const [override, setOverride] = useState('')
  const fieldId = `override-${item.row_index}-${item.family}`
  const lowConfidence = (item.confidence ?? 0) < 0.5

  return (
    <motion.li
      layout={!reduced}
      initial={reduced ? false : { opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={reduced ? undefined : { opacity: 0, height: 0 }}
      transition={{ duration: reduced ? 0 : 0.2 }}
      onFocusCapture={onFocus}
      onMouseEnter={onFocus}
      data-testid={`queue-item-${item.row_index}-${item.family}`}
      className={`rounded-lg border p-3 transition-colors ${
        item.decision
          ? 'border-emerald-400/40 bg-emerald-400/5'
          : active
            ? 'border-primary bg-primary/5'
            : 'border-border'
      }`}
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex min-w-0 flex-col gap-1">
          <span className="flex flex-wrap items-center gap-2 text-sm font-medium">
            {item.family}
            <Badge variant="outline" className="text-xs">
              {familyType}
            </Badge>
            <span className="text-xs text-muted-foreground">row {item.row_index}</span>
            {item.decision && (
              <Badge variant="outline" className="border-emerald-400/40 text-emerald-400">
                {item.decision}
                {item.decision === 'override' && ` → ${String(item.override)}`}
              </Badge>
            )}
          </span>
          <span className="flex items-center gap-2 text-sm">
            <span className="text-muted-foreground">Jev said</span>
            <span className="font-medium">{String(item.answer ?? '—')}</span>
            {item.confidence !== null && (
              <span
                className={`text-xs tabular-nums ${lowConfidence ? 'text-amber-400' : 'text-muted-foreground'}`}
              >
                {`confidence ${item.confidence.toFixed(2)}`}
              </span>
            )}
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-1.5">
          <Button size="sm" variant="outline" onClick={() => onDecide('accept')}>
            <CheckIcon aria-hidden />
            Accept
          </Button>
          <Button size="sm" variant="outline" onClick={() => onDecide('exclude')}>
            <XIcon aria-hidden />
            Exclude
          </Button>
        </div>
      </div>

      {state && (
        <details className="mt-2">
          <summary className="cursor-pointer text-xs text-muted-foreground">Show the State</summary>
          <pre className="mt-1 overflow-x-auto whitespace-pre-wrap text-xs text-muted-foreground">
            {state}
          </pre>
        </details>
      )}

      <div className="mt-2 flex flex-wrap items-end gap-2">
        <div className="flex min-w-40 flex-1 flex-col gap-1">
          <LabelField htmlFor={fieldId} className="text-xs text-muted-foreground">
            {`Override ${item.family} with`}
          </LabelField>
          <Input
            id={fieldId}
            value={override}
            placeholder={familyType === 'score' ? '5' : familyType === 'noul' ? 'yes / no' : 'an option'}
            onChange={(event) => setOverride(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && override.trim()) onDecide('override', override.trim())
            }}
          />
        </div>
        <Button
          size="sm"
          disabled={!override.trim()}
          onClick={() => onDecide('override', override.trim())}
        >
          Override
        </Button>
      </div>
    </motion.li>
  )
}

/** Row 1 of the version, for the expandable State on each card. */
function useRowPreview(versionId: string | undefined) {
  const query = useQuery({
    queryKey: ['review-row', versionId],
    queryFn: () =>
      apiGet<{ columns: Array<{ name: string }>; rows: unknown[][] }>(
        `/dataset-versions/${versionId}/preview?page_size=1`,
      ).then((body) => {
        const row: Record<string, unknown> = {}
        body.columns.forEach((column, i) => {
          row[column.name] = body.rows[0]?.[i] ?? null
        })
        return row
      }),
    enabled: Boolean(versionId),
    retry: false,
  })
  return { data: query.data ?? null }
}
