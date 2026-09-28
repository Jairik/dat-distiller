/**
 * The Label step: choose the State, write the Jev Questions, preview, run.
 *
 * The order is the point. Labeling costs a Jev call per row, so a preview has
 * to come first and the run does not unlock until one has been seen. The step
 * then ends with the Checks panel, which is what gates the way to Train.
 *
 * Writing questions by hand stays the primary path; "Draft with Provider" is an
 * optional helper that fills the builder and leaves it fully editable.
 */

import { useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { motion, useReducedMotion } from 'motion/react'
import {
  ArrowUpIcon,
  DownloadIcon,
  PlusIcon,
  SparklesIcon,
  Trash2Icon,
  TriangleAlertIcon,
} from 'lucide-react'

import { apiGet, apiPost } from '@/api/client'

import { ChecksPanel } from '@/components/checks/checks-panel'
import { JobProgress } from '@/components/jobs/job-progress'
import { StaggerGroup, StaggerItem } from '@/components/motion'
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
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import {
  QUESTION_TYPES,
  emptyQuestion,
  labelColumns,
  questionErrors,
  questionSpecOrNull,
  questionToSpec,
  serializeState,
  useLabelEstimate,
  useLabelPreview,
  useStartLabelRun,
  type JevQuestionDraft,
  type LabelPreview,
  type QuestionType,
} from '@/lib/label'
import { fmtNumber } from '@/lib/format'
import { useProjectContext } from '@/routes/project-page'

const TYPE_HELP: Record<QuestionType, string> = {
  noul: 'A yes/no judgement. Jev answers yes or no with a confidence.',
  choice: 'Exactly one of several named options, with a probability for each.',
  score: 'A level on an ordered scale, lowest first.',
}

export function LabelStep() {
  const { project, versions } = useProjectContext()
  const navigate = useNavigate()
  const [params] = useSearchParams()

  const versionId = params.get('version') ?? versions.at(-1)?.id ?? ''
  const version = versions.find((v) => v.id === versionId) ?? null

  const [stateColumns, setStateColumns] = useState<string[]>([])
  const [questions, setQuestions] = useState<JevQuestionDraft[]>([emptyQuestion(0)])

  const preview = useLabelPreview()
  const estimate = useLabelEstimate()
  const startRun = useStartLabelRun()
  const [jobId, setJobId] = useState<string | null>(null)
  const [labeledVersionId, setLabeledVersionId] = useState<string | null>(null)

  const specs = useMemo(
    () => questions.map(questionSpecOrNull).filter((s): s is Record<string, unknown> => s !== null),
    [questions],
  )
  const outputs = useMemo(() => labelColumns(specs), [specs])
  const allValid = questions.every((q) => questionErrors(q).length === 0) && questions.length > 0

  // the effective State is everything chosen, minus any Label Column we are
  // about to add — sending both would have Jev read its own output
  const effectiveState = useMemo(
    () => stateColumns.filter((column) => !outputs.includes(column)),
    [stateColumns, outputs],
  )

  const body =
    versionId && allValid && effectiveState.length > 0
      ? {
          version_id: versionId,
          questions: specs,
          state_columns: effectiveState,
        }
      : null

  const canPreview = body !== null && !preview.isPending
  const canRun = body !== null && preview.data !== null && !startRun.isPending && !jobId

  return (
    <div className="flex flex-col gap-6">
      <Card>
        <CardHeader>
          <CardTitle>Label</CardTitle>
          <CardDescription>
            Pick the State Jev reads, write your Jev Questions, and check the answers on a few
            rows before labeling the whole Dataset Version. Labeling makes one Jev call per row.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-5">
          {versions.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              No Dataset Versions yet. Generate or upload one first.
            </p>
          ) : (
            <StatePicker
              version={version}
              versionId={versionId}
              onVersion={() => {
                setStateColumns([])
                preview.reset()
              }}
              selected={stateColumns}
              outputs={outputs}
              onChange={(next) => {
                setStateColumns(next)
                preview.reset()
              }}
            />
          )}

          <QuestionBuilder
            questions={questions}
            onChange={(next) => {
              setQuestions(next)
              preview.reset()
            }}
          />
        </CardContent>
        <CardFooter className="flex flex-wrap items-center gap-3">
          <Button disabled={!canPreview} onClick={() => (body ? preview.mutate(body) : undefined)}>
            {preview.isPending ? 'Asking Jev…' : 'Preview 5 rows'}
          </Button>
          {body && !preview.data && (
            <p className="text-sm text-muted-foreground">
              Preview a few rows to unlock the full run.
            </p>
          )}
        </CardFooter>
      </Card>

      {preview.isError && (
        <p role="alert" className="flex items-center gap-2 text-sm text-destructive">
          <TriangleAlertIcon aria-hidden className="size-4" />
          {(preview.error as Error).message}
        </p>
      )}

      {preview.data && <PreviewAnswers preview={preview.data} />}

      {body && preview.data && !estimate.data && !jobId && (
        <Button variant="outline" disabled={estimate.isPending} onClick={() => estimate.mutate(body)}>
          {estimate.isPending ? 'Estimating…' : 'Estimate and continue'}
        </Button>
      )}

      {estimate.data && (
        <Card>
          <CardHeader>
            <CardTitle>Ready to label</CardTitle>
            <CardDescription>
              {fmtNumber(estimate.data.rows)} rows · about{' '}
              {fmtNumber(estimate.data.estimated_jev_calls)} Jev call(s), one per row.
            </CardDescription>
          </CardHeader>
          <CardFooter>
            <Button
              disabled={!canRun}
              onClick={() => {
                if (!body) return
                setJobId(null)
                startRun.mutate(body, { onSuccess: setJobId })
              }}
            >
              {startRun.isPending ? 'Starting…' : 'Label the whole Dataset Version'}
            </Button>
          </CardFooter>
        </Card>
      )}

      {startRun.isError && (
        <p role="alert" className="text-sm text-destructive">
          {(startRun.error as Error).message}
        </p>
      )}

      {jobId && (
        <RunPanel
          jobId={jobId}
          onFinished={(result) => {
            if (result?.version_id) setLabeledVersionId(String(result.version_id))
          }}
          onAgain={() => {
            setJobId(null)
            preview.reset()
          }}
          onOpen={(id) => navigate(`/projects/${project.id}/dataset?v=${id}`)}
        />
      )}

      {labeledVersionId && (
        <div className="flex flex-wrap gap-2">
          <Button asChild variant="outline" size="sm">
            <a href={`/api/dataset-versions/${labeledVersionId}/download?format=csv`} download>
              <DownloadIcon aria-hidden />
              Download CSV
            </a>
          </Button>
        </div>
      )}

      {labeledVersionId && (
        <ChecksPanel
          subjectType="dataset_version"
          subjectId={labeledVersionId}
          continueLabel="Continue to Train"
          onContinue={() => navigate(`/projects/${project.id}/train?version=${labeledVersionId}`)}
        />
      )}
    </div>
  )
}

// -- the State picker ---------------------------------------------------------

function StatePicker({
  version,
  versionId,
  onVersion,
  selected,
  outputs,
  onChange,
}: {
  version: { id: string; number: number; row_count: number; origin: string } | null
  versionId: string
  onVersion: (id: string) => void
  selected: string[]
  outputs: string[]
  onChange: (next: string[]) => void
}) {
  const { versions } = useProjectContext()
  const sample = useSampleRow(versionId)
  const all = sample.data?.columns ?? []
  // columns this Labeling run is about to create cannot be part of its own input
  const available = all.filter((column) => !outputs.includes(column))

  return (
    <fieldset className="flex flex-col gap-3">
      <legend className="text-sm font-medium">State</legend>
      <p className="text-xs text-muted-foreground">
        The State is one row written out as &ldquo;column: value&rdquo; lines — exactly what Jev
        reads. Leave a column out and Jev never sees it.
      </p>

      <div className="flex flex-col gap-1.5">
        <Label htmlFor="label-version">Dataset Version</Label>
        <select
          id="label-version"
          className={SELECT_CLASS}
          value={versionId}
          onChange={(event) => onVersion(event.target.value)}
        >
          <option value="">Choose a version…</option>
          {versions.map((v) => (
            <option key={v.id} value={v.id}>
              v{v.number} · {fmtNumber(v.row_count)} rows · {v.origin}
            </option>
          ))}
        </select>
      </div>

      {available.length > 0 ? (
        <div className="flex flex-wrap items-center gap-2">
          <Button
            size="sm"
            variant="ghost"
            onClick={() => onChange(selected.length === available.length ? [] : available)}
          >
            {selected.length === available.length ? 'Clear all' : 'Select every column'}
          </Button>
          <span className="text-xs text-muted-foreground">
            {selected.length} of {available.length} column(s) selected
          </span>
        </div>
      ) : (
        <p className="text-xs text-muted-foreground">
          {versionId ? 'Reading the columns of this Dataset Version…' : 'Choose a Dataset Version.'}
        </p>
      )}

      <ul className="grid max-h-48 gap-1 overflow-y-auto sm:grid-cols-2">
        {available.map((column) => (
          <li key={column} className="flex items-center gap-2 text-sm">
            <Checkbox
              id={`state-${column}`}
              checked={selected.includes(column)}
              onCheckedChange={(checked) =>
                onChange(
                  checked === true
                    ? [...selected, column]
                    : selected.filter((c) => c !== column),
                )
              }
            />
            <Label htmlFor={`state-${column}`} className="font-normal">
              {column}
            </Label>
          </li>
        ))}
      </ul>

      {selected.length > 0 && (
        <div className="flex flex-col gap-1">
          <span className="text-xs font-medium text-muted-foreground">
            What Jev sees for row 1 of v{version?.number ?? '?'}:
          </span>
          <pre
            data-testid="state-preview"
            className="overflow-x-auto whitespace-pre-wrap rounded-md border border-border bg-muted/40 p-2 font-mono text-xs"
          >
            {sample.data
              ? serializeState(sample.data.row, selected)
              : 'Pick a Dataset Version to see its State.'}
          </pre>
        </div>
      )}
    </fieldset>
  )
}

const SELECT_CLASS =
  'h-9 w-full min-w-0 rounded-md border border-input bg-transparent px-3 py-1 text-base shadow-xs outline-none transition-[color,box-shadow] focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 disabled:pointer-events-none disabled:opacity-50 md:text-sm dark:bg-input/30'

/** Row 1 of a version, for the live State preview. */
function useSampleRow(versionId: string) {
  const query = useQuery({
    queryKey: ['dataset-version-row', versionId],
    queryFn: () =>
      apiGet<{ columns: Array<{ name: string }>; rows: unknown[][] }>(
        `/dataset-versions/${versionId}/preview?page_size=1`,
      ).then((body) => {
        const columns = body.columns.map((c) => c.name)
        const row: Record<string, unknown> = {}
        columns.forEach((name, i) => {
          row[name] = body.rows[0]?.[i] ?? null
        })
        return { columns, row }
      }),
    enabled: Boolean(versionId),
    retry: false,
  })
  return { data: query.data ?? null }
}

// -- the Jev Question builder -------------------------------------------------

function QuestionBuilder({
  questions,
  onChange,
}: {
  questions: JevQuestionDraft[]
  onChange: (next: JevQuestionDraft[]) => void
}) {
  const update = (index: number, patch: Partial<JevQuestionDraft>) =>
    onChange(questions.map((q, i) => (i === index ? { ...q, ...patch } : q)))

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-sm font-medium">Jev Questions</span>
        <Button size="sm" variant="ghost" onClick={() => onChange([...questions, emptyQuestion(questions.length)])}>
          <PlusIcon aria-hidden />
          Add a question
        </Button>
      </div>

      <StaggerGroup key={questions.length}>
        {questions.map((question, index) => (
          <StaggerItem key={index}>
            <QuestionEditor
              question={question}
              index={index}
              removable={questions.length > 1}
              onChange={(patch) => update(index, patch)}
              onRemove={() => onChange(questions.filter((_, i) => i !== index))}
            />
          </StaggerItem>
        ))}
      </StaggerGroup>
    </div>
  )
}

function QuestionEditor({
  question,
  index,
  removable,
  onChange,
  onRemove,
}: {
  question: JevQuestionDraft
  index: number
  removable: boolean
  onChange: (patch: Partial<JevQuestionDraft>) => void
  onRemove: () => void
}) {
  const draft = useDraftQuestion(question, onChange)
  const errors = questionErrors(question)
  const previewText = JSON.stringify(questionToSpec(question), null, 2)

  return (
    <div className="flex flex-col gap-3 rounded-lg border border-border p-3">
      <div className="flex flex-wrap items-start gap-2">
        <div className="flex min-w-40 flex-1 flex-col gap-1.5">
          <Label htmlFor={`q-name-${index}`} className="text-xs text-muted-foreground">
            Name (becomes the Label Column)
          </Label>
          <Input
            id={`q-name-${index}`}
            value={question.name}
            onChange={(event) => onChange({ name: event.target.value })}
          />
        </div>
        <Button size="sm" variant="outline" disabled={draft.isPending} onClick={draft.draft}>
          <SparklesIcon aria-hidden />
          {draft.isPending ? 'Drafting…' : 'Draft with Provider'}
        </Button>
        {removable && (
          <Button
            size="icon"
            variant="ghost"
            aria-label={`Remove question ${index + 1}`}
            onClick={onRemove}
          >
            <Trash2Icon aria-hidden />
          </Button>
        )}
      </div>

      <fieldset className="flex flex-col gap-2">
        <legend className="text-xs text-muted-foreground">Type</legend>
        <div className="flex flex-wrap gap-2">
          {QUESTION_TYPES.map((type) => (
            <label
              key={type}
              className={`flex cursor-pointer items-center gap-1.5 rounded-md border px-2 py-1 text-sm ${
                question.type === type ? 'border-primary bg-primary/5' : 'border-border'
              }`}
            >
              <input
                type="radio"
                name={`q-type-${index}`}
                className="size-3.5 accent-primary"
                checked={question.type === type}
                onChange={() => onChange({ type })}
              />
              {type}
            </label>
          ))}
        </div>
        <p className="text-xs text-muted-foreground">{TYPE_HELP[question.type]}</p>
      </fieldset>

      <div className="flex flex-col gap-1.5">
        <Label htmlFor={`q-instructions-${index}`} className="text-xs text-muted-foreground">
          Instructions
        </Label>
        <Textarea
          id={`q-instructions-${index}`}
          rows={2}
          placeholder="Answer yes when the ticket clearly asks for a refund; otherwise answer no."
          value={question.instructions}
          onChange={(event) => onChange({ instructions: event.target.value })}
        />
      </div>

      {question.type === 'choice' && (
        <div className="flex flex-col gap-2">
          <span className="text-xs text-muted-foreground">
            Options — the key is what lands in the Label Column, the description is what Jev reads
          </span>
          {question.criteria.map((criterion, ci) => (
            <div key={ci} className="flex flex-wrap items-end gap-2">
              <div className="flex min-w-32 flex-1 flex-col gap-1">
                <Label htmlFor={`q-crit-key-${index}-${ci}`} className="text-[11px] text-muted-foreground">
                  Key
                </Label>
                <Input
                  id={`q-crit-key-${index}-${ci}`}
                  value={criterion.key}
                  onChange={(event) =>
                    onChange({
                      criteria: question.criteria.map((c, i) =>
                        i === ci ? { ...c, key: event.target.value } : c,
                      ),
                    })
                  }
                />
              </div>
              <div className="flex min-w-48 flex-[2] flex-col gap-1">
                <Label htmlFor={`q-crit-desc-${index}-${ci}`} className="text-[11px] text-muted-foreground">
                  Description
                </Label>
                <Input
                  id={`q-crit-desc-${index}-${ci}`}
                  value={criterion.description}
                  placeholder="about invoices, charges or refunds"
                  onChange={(event) =>
                    onChange({
                      criteria: question.criteria.map((c, i) =>
                        i === ci ? { ...c, description: event.target.value } : c,
                      ),
                    })
                  }
                />
              </div>
              {question.criteria.length > 2 && (
                <Button
                  size="icon"
                  variant="ghost"
                  aria-label={`Remove option ${criterion.key || ci + 1}`}
                  onClick={() =>
                    onChange({ criteria: question.criteria.filter((_, i) => i !== ci) })
                  }
                >
                  <Trash2Icon aria-hidden />
                </Button>
              )}
            </div>
          ))}
          <Button
            size="sm"
            variant="ghost"
            className="self-start"
            onClick={() =>
              onChange({
                criteria: [
                  ...question.criteria,
                  { key: `option_${question.criteria.length + 1}`, description: '' },
                ],
              })
            }
          >
            <PlusIcon aria-hidden />
            Add an option
          </Button>
        </div>
      )}

      {question.type === 'score' && (
        <div className="flex flex-col gap-2">
          <span className="text-xs text-muted-foreground">
            Levels, lowest to highest — the order is the scale
          </span>
          {question.levels.map((level, li) => (
            <div key={li} className="flex items-end gap-2">
              <div className="flex flex-1 flex-col gap-1">
                <Label htmlFor={`q-level-${index}-${li}`} className="text-[11px] text-muted-foreground">
                  Level {li + 1}
                </Label>
                <Input
                  id={`q-level-${index}-${li}`}
                  value={level}
                  placeholder={li === 0 ? '1 - unusable' : '5 - publication ready'}
                  onChange={(event) =>
                    onChange({
                      levels: question.levels.map((l, i) =>
                        i === li ? event.target.value : l,
                      ),
                    })
                  }
                />
              </div>
              {question.levels.length > 2 && (
                <Button
                  size="icon"
                  variant="ghost"
                  aria-label={`Remove level ${li + 1}`}
                  onClick={() => onChange({ levels: question.levels.filter((_, i) => i !== li) })}
                >
                  <Trash2Icon aria-hidden />
                </Button>
              )}
            </div>
          ))}
          <Button
            size="sm"
            variant="ghost"
            className="self-start"
            onClick={() => onChange({ levels: [...question.levels, ''] })}
          >
            <PlusIcon aria-hidden />
            Add a level
          </Button>
        </div>
      )}

      <details className="text-xs">
        <summary className="cursor-pointer text-muted-foreground">What gets sent to Jev</summary>
        <pre className="mt-1 overflow-x-auto rounded-md border border-border bg-muted/40 p-2">
          {previewText}
        </pre>
      </details>

      {draft.isError && (
        <p role="alert" className="text-xs text-destructive">
          {(draft.error as Error).message}
        </p>
      )}
      {errors.length > 0 && (
        <ul role="alert" className="flex flex-col gap-0.5 text-xs text-amber-400">
          {errors.map((error) => (
            <li key={error}>{error}</li>
          ))}
        </ul>
      )}
    </div>
  )
}

/**
 * The optional "Draft with Provider" helper.
 *
 * It *fills* the builder and then gets out of the way: the result arrives
 * through `onDraft` as ordinary state, so every field stays editable and the
 * user is never locked into what a Provider suggested.
 */
function useDraftQuestion(
  seed: JevQuestionDraft,
  onDraft: (question: JevQuestionDraft) => void,
) {
  const mutation = useMutation({
    mutationFn: (draft: JevQuestionDraft) =>
      apiPost<{ question: Record<string, unknown> }>('/jev/draft-question', {
        description: draft.instructions.trim() || draft.name.trim() || 'a dataset to label',
      }).then((body) => fromDraftedQuestion(body.question)),
    onSuccess: onDraft,
  })
  return {
    draft: () => mutation.mutate(seed),
    isPending: mutation.isPending,
    isError: mutation.isError,
    error: (mutation.error as Error | null) ?? null,
  }
}

/** Turn a Provider's draft back into an editable builder draft. */
function fromDraftedQuestion(raw: Record<string, unknown>): JevQuestionDraft {
  const criteria = (raw.criteria ?? {}) as Record<string, string>
  const levels = raw.levels
  return {
    type: (QUESTION_TYPES as readonly string[]).includes(String(raw.type))
      ? (raw.type as QuestionType)
      : 'noul',
    name: String(raw.name ?? ''),
    instructions: String(raw.instructions ?? ''),
    criteria:
      Object.keys(criteria).length > 0
        ? Object.entries(criteria).map(([key, description]) => ({
            key,
            description: String(description ?? ''),
          }))
        : emptyQuestion(0).criteria,
    levels: Array.isArray(levels) && levels.length > 0 ? levels.map(String) : emptyQuestion(0).levels,
  }
}

// -- preview + run ------------------------------------------------------------

function PreviewAnswers({ preview }: { preview: LabelPreview }) {
  const reduced = useReducedMotion()
  const names = Object.keys(preview.rows[0]?.answers ?? {})
  return (
    <Card>
      <CardHeader>
        <CardTitle>Preview</CardTitle>
        <CardDescription>
          {preview.rows.length} rows · {preview.state_columns.length} State column(s). Nothing is
          saved yet — this is what Jev will say about the whole Dataset Version.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <motion.div
          initial={reduced ? false : { opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: reduced ? 0 : 0.24 }}
          className="flex flex-col gap-3"
        >
          {preview.rows.map((row, i) => (
            <div key={i} className="rounded-lg border border-border p-3">
              <pre className="overflow-x-auto whitespace-pre-wrap text-xs text-muted-foreground">
                {row.state}
              </pre>
              <ul className="mt-2 flex flex-col gap-1">
                {names.map((name) => {
                  const answer = row.answers[name]
                  return (
                    <li key={name} className="flex flex-wrap items-center gap-2 text-sm">
                      <Badge variant="outline">{name}</Badge>
                      <span className="font-medium">{String(answer.answer ?? '—')}</span>
                      {answer.confidence !== undefined && (
                        <span className="text-xs tabular-nums text-muted-foreground">
                          {`confidence ${answer.confidence.toFixed(2)}`}
                        </span>
                      )}
                    </li>
                  )
                })}
              </ul>
            </div>
          ))}
        </motion.div>
      </CardContent>
    </Card>
  )
}

function RunPanel({
  jobId,
  onFinished,
  onAgain,
  onOpen,
}: {
  jobId: string
  onFinished: (result: Record<string, unknown> | null) => void
  onAgain: () => void
  onOpen: (versionId: string) => void
}) {
  // JobProgress already holds the SSE stream; taking the status from it avoids
  // opening a second connection to the same job
  const [status, setStatus] = useState<string | null>(null)
  const resume = useResume(jobId, status)
  return (
    <Card>
      <CardHeader>
        <CardTitle>Labeling run</CardTitle>
        <CardDescription>
          One Jev call per row. The run checkpoints as it goes, so an interruption is resumable.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <JobProgress
          jobId={jobId}
          label="Labeling"
          onSettled={onFinished}
          onStatus={setStatus}
          terminalExtra={
            resume.available ? (
              <div className="mt-2 flex flex-col gap-2">
                <p className="text-sm text-muted-foreground">
                  This run was interrupted. Rows already labeled are kept — resuming only pays for
                  what is left.
                </p>
                <Button size="sm" onClick={resume.run}>
                  <ArrowUpIcon aria-hidden />
                  Resume the run
                </Button>
                {resume.error && (
                  <p role="alert" className="text-sm text-destructive">
                    {resume.error.message}
                  </p>
                )}
              </div>
            ) : null
          }
          renderResult={(result) => (
            <div className="flex flex-col gap-2">
              <p className="text-sm">
                {`${fmtNumber(Number(result.labeled_rows ?? 0))} rows labeled`}
                {Number(result.failed_count ?? 0) > 0 && (
                  <span className="text-amber-400">
                    {` · ${String(result.failed_count)} row(s) Jev could not answer`}
                  </span>
                )}
              </p>
              <ul className="flex flex-wrap gap-1.5">
                {(result.label_columns as string[] | undefined)?.map((column) => (
                  <li key={column}>
                    <Badge variant="outline" className="font-mono text-xs">
                      {column}
                    </Badge>
                  </li>
                ))}
              </ul>
              <div className="flex flex-wrap gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => onOpen(String(result.version_id))}
                >
                  Open the new Dataset Version
                </Button>
                <Button size="sm" variant="ghost" onClick={onAgain}>
                  Label again
                </Button>
              </div>
            </div>
          )}
        />
      </CardContent>
    </Card>
  )
}

/**
 * Resume is offered only when the job says it is resumable and has actually
 * stopped — an interrupted Labeling run should cost nothing extra to pick up.
 */
function useResume(jobId: string | null, status: string | null) {
  const queryClient = useQueryClient()
  const mutation = useMutation({
    mutationFn: (id: string) => apiPost<{ id: string }>(`/jobs/${id}/resume`, {}),
    onSuccess: () => {
      // resuming re-queues the job, so the snapshot has to start listening again
      if (jobId) queryClient.invalidateQueries({ queryKey: ['job', jobId] })
    },
  })
  return {
    available: Boolean(jobId) && status === 'interrupted',
    run: () => {
      if (jobId) mutation.mutate(jobId)
    },
    error: (mutation.error as Error | null) ?? null,
  }
}
