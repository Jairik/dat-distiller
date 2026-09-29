/**
 * The Generate step: describe what you want, preview it, then run it.
 *
 * The flow is deliberately gated in one direction only. Preview comes first and
 * the run button stays shut until a preview exists, because a Generation Run is
 * a Provider call that costs money — you should have seen the rows first. The
 * step then ends with the Checks panel, which is what gates the way to Label.
 */

import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, useReducedMotion } from 'motion/react'
import { PlusIcon, SparklesIcon, Trash2Icon, TriangleAlertIcon } from 'lucide-react'

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
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import {
  MODES,
  SPEC_TYPES,
  emptySpec,
  localSpecErrors,
  toSpec,
  toWireSpec,
  useEstimate,
  usePreview,
  useProviderOptions,
  useStartRun,
  useSuggestColumns,
  type ColumnSpec,
  type GenerationBody,
  type GenerationMode,
  type SpecType,
} from '@/lib/generate'
import { fmtNumber } from '@/lib/format'
import { useProjectContext } from '@/routes/project-page'

const SELECT_CLASS =
  'h-9 w-full min-w-0 rounded-md border border-input bg-transparent px-3 py-1 text-base shadow-xs outline-none transition-[color,box-shadow] focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 disabled:pointer-events-none disabled:opacity-50 md:text-sm dark:bg-input/30'

/** Plain-language mode explanations — the names alone do not say what happens. */
const MODE_HELP: Record<GenerationMode, string> = {
  hybrid:
    'Draws the numbers from a Gaussian copula fitted to your data, then asks the Provider to write the free-text columns. The default, and the best mix of realism and cost.',
  statistical:
    'Pure statistics — no Provider calls at all. Fast and free, but text columns come out empty.',
  llm:
    'Asks the Provider for every row. Most realistic prose, but every row costs a Provider call and nothing is guaranteed to match your distributions.',
}

type Source = 'sample' | 'specs'

export function GenerateStep() {
  const { project, versions, reloadVersions } = useProjectContext()
  const navigate = useNavigate()

  const [description, setDescription] = useState('')
  const [source, setSource] = useState<Source>('sample')
  const [sampleVersionId, setSampleVersionId] = useState('')
  const [specs, setSpecs] = useState<ColumnSpec[]>([emptySpec(0)])
  const [mode, setMode] = useState<GenerationMode>('hybrid')
  const [count, setCount] = useState(500)
  const [seed, setSeed] = useState('')
  const [balanceColumn, setBalanceColumn] = useState('')
  const [balanceShare, setBalanceShare] = useState('')
  const [provider, setProvider] = useState('')

  const preview = usePreview()
  const estimate = useEstimate()
  const startRun = useStartRun()
  const [jobId, setJobId] = useState<string | null>(null)
  const [ranVersionId, setRanVersionId] = useState<string | null>(null)

  /**
   * Anything that changes what will be run invalidates both the preview and the
   * estimate.
   *
   * They used to be separate concerns handled by whoever remembered: only
   * `description`, `source`, `sample` and `specs` reset the preview, and the
   * estimate was never reset at all. So the panel could answer "what will this
   * cost me" for a different Generation Mode, row count, Provider and Balance
   * Target than the one about to be sent, and the "Estimate and continue"
   * button that would refresh it is hidden whenever an estimate exists. The
   * estimate describes a *run*, exactly as the preview does, so it is
   * invalidated the same way and by the same call.
   */
  function invalidate() {
    preview.reset()
    estimate.reset()
  }

  const specErrors = source === 'specs' ? localSpecErrors(specs) : []
  const body: GenerationBody | null = useMemo(() => {
    if (description.trim().length < 3) return null
    if (source === 'sample' && !sampleVersionId) return null
    if (source === 'specs' && (specs.length === 0 || specErrors.length > 0)) return null
    return {
      project_id: project.id,
      description: description.trim(),
      mode,
      count,
      ...(source === 'sample'
        ? { sample_version_id: sampleVersionId }
        : { specs: specs.map((spec) => toWireSpec(spec) as never) }),
      ...(balanceColumn.trim() && Number(balanceShare) > 0
        ? { balance: { [balanceColumn.trim()]: { positive: Number(balanceShare) } } }
        : {}),
      ...(provider ? { provider } : {}),
      ...(seed.trim() ? { seed: Number(seed) } : {}),
    }
  }, [
    description,
    project.id,
    source,
    sampleVersionId,
    specs,
    specErrors.length,
    mode,
    count,
    balanceColumn,
    balanceShare,
    provider,
    seed,
  ])

  // Preview is the gate: no successful preview, no run.
  const canPreview = body !== null && !preview.isPending
  const canRun = body !== null && preview.data !== null && !startRun.isPending && !jobId

  function onStarted() {
    setRanVersionId(null)
    preview.reset()
  }

  const done = ranVersionId !== null

  return (
    <div className="flex flex-col gap-6">
      <Card>
        <CardHeader>
          <CardTitle>Generate</CardTitle>
          <CardDescription>
            Describe the dataset you want. Dat Distiller fits it to your data, previews a few rows
            for you to check, and only then spends Provider calls on the full run.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-5">
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="generate-description">What should this dataset contain?</Label>
            <Textarea
              id="generate-description"
              rows={3}
              value={description}
              placeholder="Support tickets for a SaaS help desk, with a topic, a sentiment and a resolution time."
              onChange={(event) => {
                setDescription(event.target.value)
                invalidate()
              }}
            />
          </div>

          <SourceChoice
            source={source}
            onChange={(next) => {
              setSource(next)
              invalidate()
            }}
          />

          {source === 'sample' ? (
            <SamplePicker
              versions={versions}
              value={sampleVersionId}
              onChange={(id) => {
                setSampleVersionId(id)
                invalidate()
              }}
            />
          ) : (
            <SpecTable
              specs={specs}
              errors={specErrors}
              description={description}
              onChange={(next) => {
                setSpecs(next)
                invalidate()
              }}
            />
          )}

          <ModePicker value={mode} onChange={(next) => { setMode(next); invalidate() }} />

          <div className="grid gap-4 sm:grid-cols-3">
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="generate-count">Rows</Label>
              <Input
                id="generate-count"
                type="number"
                min={1}
                max={100000}
                value={count}
                onChange={(event) => { setCount(Math.max(1, Number(event.target.value) || 1)); invalidate() }}
              />
              <p className="text-xs text-muted-foreground">
                {fmtNumber(count)} rows. A soft limit raises a Check, it never blocks.
              </p>
            </div>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="generate-seed">Seed (optional)</Label>
              <Input
                id="generate-seed"
                type="number"
                placeholder="leave blank to pick one"
                value={seed}
                onChange={(event) => { setSeed(event.target.value); invalidate() }}
              />
              <p className="text-xs text-muted-foreground">
                The same seed reproduces the same rows.
              </p>
            </div>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="generate-provider">Provider (optional)</Label>
              <ProviderPicker value={provider} onChange={(next) => { setProvider(next); invalidate() }} />
              <p className="text-xs text-muted-foreground">
                Defaults to the one chosen in Settings.
              </p>
            </div>
          </div>

          <BalanceFields
            column={balanceColumn}
            share={balanceShare}
            onColumn={(next) => { setBalanceColumn(next); invalidate() }}
            onShare={(next) => { setBalanceShare(next); invalidate() }}
          />
        </CardContent>
        <CardFooter className="flex flex-wrap items-center gap-3">
          <Button disabled={!canPreview} onClick={() => (body ? preview.mutate(body) : undefined)}>
            {preview.isPending ? 'Previewing…' : 'Preview 5 rows'}
          </Button>
          {!body && (
            <p className="text-sm text-muted-foreground">
              {description.trim().length < 3
                ? 'Describe the dataset first.'
                : source === 'sample'
                  ? 'Pick a sample Dataset Version.'
                  : 'Fix the Column Specs first.'}
            </p>
          )}
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

      {preview.data && <PreviewTable rows={preview.data.rows} seed={preview.data.seed} />}

      {estimate.data && preview.data && (
        <EstimatePanel
          estimate={estimate.data}
          onConfirm={() => {
            if (!body) return
            setJobId(null)
            startRun.mutate(body, { onSuccess: setJobId })
          }}
          running={Boolean(jobId)}
          canRun={canRun}
        />
      )}

      {preview.data && body && !jobId && !estimate.data && (
        <Button
          variant="outline"
          onClick={() => estimate.mutate(body)}
          disabled={estimate.isPending}
        >
          {estimate.isPending ? 'Estimating…' : 'Estimate and continue'}
        </Button>
      )}

      {startRun.isError && (
        <p role="alert" className="text-sm text-destructive">
          {(startRun.error as Error).message}
        </p>
      )}

      {jobId && (
        <Card>
          <CardHeader>
            <CardTitle>Generation Run</CardTitle>
            <CardDescription>
              This is a background job — you can leave this page and come back to it.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <JobProgress
              jobId={jobId}
              label="Generating"
              onSettled={(result) => {
                if (result?.version_id) {
                  setRanVersionId(String(result.version_id))
                  reloadVersions()
                }
              }}
              key={jobId}
              renderResult={(result) => (
                <div className="flex flex-col gap-2">
                  <p className="text-sm">
                    {fmtNumber(Number(result.rows ?? 0))} rows · seed{' '}
                    <span className="font-mono">{String(result.seed)}</span> ·{' '}
                    {Number(result.provider_calls ?? 0)} Provider call(s)
                  </p>
                  {Number(result.dropped_rows ?? 0) > 0 && (
                    <p className="text-sm text-muted-foreground">
                      {fmtNumber(Number(result.dropped_rows))} rows were dropped because the
                      Provider could not fit them to the fitted data.
                    </p>
                  )}
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() =>
                      navigate(`/projects/${project.id}/dataset?v=${String(result.version_id)}`)
                    }
                  >
                    Open the new Dataset Version
                  </Button>
                  <Button size="sm" variant="ghost" onClick={onStarted}>
                    Generate again
                  </Button>
                </div>
              )}
            />
          </CardContent>
        </Card>
      )}

      {done && ranVersionId && (
        <ChecksPanel
          subjectType="dataset_version"
          subjectId={ranVersionId}
          continueLabel="Continue to Label"
          onContinue={() => navigate(`/projects/${project.id}/label?version=${ranVersionId}`)}
        />
      )}
    </div>
  )
}

// -- source choice ------------------------------------------------------------

function SourceChoice({
  source,
  onChange,
}: {
  source: Source
  onChange: (next: Source) => void
}) {
  const options: Array<{ value: Source; title: string; blurb: string }> = [
    {
      value: 'sample',
      title: 'Use a sample Dataset Version',
      blurb: 'Fit to real data you already have. Best fidelity.',
    },
    {
      value: 'specs',
      title: 'Define the columns',
      blurb: 'Describe the shape yourself. Good for data you do not have yet.',
    },
  ]
  return (
    <fieldset className="flex flex-col gap-2">
      <legend className="text-sm font-medium">Where does the shape come from?</legend>
      <div className="grid gap-2 sm:grid-cols-2">
        {options.map((option) => (
          <label
            key={option.value}
            className={`flex cursor-pointer flex-col gap-0.5 rounded-lg border p-3 text-sm transition-colors ${
              source === option.value
                ? 'border-primary bg-primary/5'
                : 'border-border hover:border-primary/40'
            }`}
          >
            <span className="flex items-center gap-2 font-medium">
              <input
                type="radio"
                name="generate-source"
                className="size-4 accent-primary"
                checked={source === option.value}
                onChange={() => onChange(option.value)}
              />
              {option.title}
            </span>
            <span className="text-xs text-muted-foreground">{option.blurb}</span>
          </label>
        ))}
      </div>
    </fieldset>
  )
}

function SamplePicker({
  versions,
  value,
  onChange,
}: {
  versions: Array<{ id: string; number: number; row_count: number; origin: string }>
  value: string
  onChange: (id: string) => void
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <Label htmlFor="sample-version">Sample Dataset Version</Label>
      {versions.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          No Dataset Versions yet. Upload one on the Dataset tab, or switch to defining the columns.
        </p>
      ) : (
        <select
          id="sample-version"
          className={SELECT_CLASS}
          value={value}
          onChange={(event) => onChange(event.target.value)}
        >
          <option value="">Choose a version…</option>
          {versions.map((version) => (
            <option key={version.id} value={version.id}>
              v{version.number} · {fmtNumber(version.row_count)} rows · {version.origin}
            </option>
          ))}
        </select>
      )}
    </div>
  )
}

// -- the Column Specs table ---------------------------------------------------

function SpecTable({
  specs,
  errors,
  description,
  onChange,
}: {
  specs: ColumnSpec[]
  errors: string[]
  description: string
  onChange: (next: ColumnSpec[]) => void
}) {
  const suggest = useSuggestColumns()

  const update = (index: number, field: string, value: unknown) => {
    const next = specs.map((spec, i) =>
      i === index ? toSpec({ ...spec, [field]: value }, i) : spec,
    )
    onChange(next)
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <Label>Column Specs</Label>
        <div className="flex gap-2">
          <Button
            size="sm"
            variant="outline"
            disabled={description.trim().length < 3 || suggest.isPending}
            onClick={() =>
              suggest.mutate(
                { description: description.trim(), column_names: specs.map((s) => s.name) },
                { onSuccess: (result) => onChange(result.specs.map((spec, i) => toSpec(spec, i))) },
              )
            }
          >
            <SparklesIcon aria-hidden />
            {suggest.isPending ? 'Asking…' : 'Suggest with Provider'}
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => onChange([...specs, emptySpec(specs.length)])}
          >
            <PlusIcon aria-hidden />
            Add column
          </Button>
        </div>
      </div>

      <p className="text-xs text-muted-foreground">
        Nothing here is generated until you preview and run — these are just instructions.
      </p>

      {suggest.isError && (
        <p role="alert" className="text-xs text-destructive">
          {(suggest.error as Error).message}
        </p>
      )}

      <ul className="flex flex-col gap-2">
        <StaggerGroup key={`${specs.length}-${suggest.data ? 'suggested' : 'manual'}`}>
          {specs.map((spec, index) => (
            <StaggerItem key={index}>
              <li className="grid gap-2 rounded-lg border border-border p-2 sm:grid-cols-[1fr_8rem_6rem_6rem_auto]">
                <div className="flex flex-col gap-1">
                  <Label htmlFor={`spec-name-${index}`} className="text-xs text-muted-foreground">
                    Name
                  </Label>
                  <Input
                    id={`spec-name-${index}`}
                    value={spec.name}
                    onChange={(event) => update(index, 'name', event.target.value)}
                  />
                </div>
                <div className="flex flex-col gap-1">
                  <Label htmlFor={`spec-type-${index}`} className="text-xs text-muted-foreground">
                    Type
                  </Label>
                  <select
                    id={`spec-type-${index}`}
                    className={SELECT_CLASS}
                    value={spec.type}
                    onChange={(event) => update(index, 'type', event.target.value as SpecType)}
                  >
                    {SPEC_TYPES.map((type) => (
                      <option key={type} value={type}>
                        {type}
                      </option>
                    ))}
                  </select>
                </div>
                {spec.type === 'number' || spec.type === 'integer' ? (
                  <>
                    <div className="flex flex-col gap-1">
                      <Label
                        htmlFor={`spec-min-${index}`}
                        className="text-xs text-muted-foreground"
                      >
                        Min
                      </Label>
                      <Input
                        id={`spec-min-${index}`}
                        type="number"
                        value={spec.min ?? ''}
                        onChange={(event) => update(index, 'min', event.target.value)}
                      />
                    </div>
                    <div className="flex flex-col gap-1">
                      <Label
                        htmlFor={`spec-max-${index}`}
                        className="text-xs text-muted-foreground"
                      >
                        Max
                      </Label>
                      <Input
                        id={`spec-max-${index}`}
                        type="number"
                        value={spec.max ?? ''}
                        onChange={(event) => update(index, 'max', event.target.value)}
                      />
                    </div>
                  </>
                ) : (
                  <div className="flex flex-col gap-1 sm:col-span-3">
                    <Label
                      htmlFor={`spec-cats-${index}`}
                      className="text-xs text-muted-foreground"
                    >
                      Categories (comma separated)
                    </Label>
                    <Input
                      id={`spec-cats-${index}`}
                      value={(spec.categories ?? []).join(', ')}
                      onChange={(event) => update(index, 'categories', event.target.value)}
                    />
                  </div>
                )}
                <Button
                  size="icon"
                  variant="ghost"
                  aria-label={`Remove ${spec.name}`}
                  className="self-end"
                  onClick={() => onChange(specs.filter((_, i) => i !== index))}
                >
                  <Trash2Icon aria-hidden />
                </Button>
              </li>
            </StaggerItem>
          ))}
        </StaggerGroup>
      </ul>

      {errors.length > 0 && (
        <ul role="alert" className="flex flex-col gap-0.5 text-xs text-destructive">
          {errors.map((error) => (
            <li key={error}>{error}</li>
          ))}
        </ul>
      )}
    </div>
  )
}

// -- mode, provider, balance --------------------------------------------------

function ModePicker({
  value,
  onChange,
}: {
  value: GenerationMode
  onChange: (next: GenerationMode) => void
}) {
  return (
    <fieldset className="flex flex-col gap-2">
      <legend className="text-sm font-medium">Generation Mode</legend>
      <div className="grid gap-2 sm:grid-cols-3">
        {MODES.map((candidate) => (
          <label
            key={candidate}
            className={`flex cursor-pointer flex-col gap-1 rounded-lg border p-3 text-sm transition-colors ${
              value === candidate ? 'border-primary bg-primary/5' : 'border-border hover:border-primary/40'
            }`}
          >
            <span className="flex items-center gap-2 font-medium">
              <input
                type="radio"
                name="generate-mode"
                className="size-4 accent-primary"
                checked={value === candidate}
                onChange={() => onChange(candidate)}
              />
              {candidate}
              {candidate === 'hybrid' && <Badge variant="outline">default</Badge>}
            </span>
            <span className="text-xs text-muted-foreground">{MODE_HELP[candidate]}</span>
          </label>
        ))}
      </div>
    </fieldset>
  )
}

function ProviderPicker({ value, onChange }: { value: string; onChange: (id: string) => void }) {
  const options = useProviderOptions()
  return (
    <select
      id="generate-provider"
      className={SELECT_CLASS}
      value={value}
      onChange={(event) => onChange(event.target.value)}
    >
      <option value="">agent default</option>
      {(options.data ?? []).map((option) => (
        <option key={option.id} value={option.id} disabled={!option.available}>
          {option.id}
          {option.available ? '' : ` — ${option.reason ?? 'unavailable'}`}
        </option>
      ))}
    </select>
  )
}

function BalanceFields({
  column,
  share,
  onColumn,
  onShare,
}: {
  column: string
  share: string
  onColumn: (value: string) => void
  onShare: (value: string) => void
}) {
  const [open, setOpen] = useState(false)
  return (
    <div className="flex flex-col gap-2">
      <Button size="sm" variant="ghost" className="self-start" onClick={() => setOpen(!open)}>
        {open ? 'Hide Balance Targets' : 'Add Balance Targets'}
      </Button>
      {open && (
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="balance-column">Categorical column</Label>
            <Input
              id="balance-column"
              value={column}
              placeholder="topic"
              onChange={(event) => onColumn(event.target.value)}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="balance-share">Share for &ldquo;positive&rdquo; (0–1)</Label>
            <Input
              id="balance-share"
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={share}
              placeholder="0.5"
              onChange={(event) => onShare(event.target.value)}
            />
          </div>
          <p className="text-xs text-muted-foreground sm:col-span-2">
            Generation will try to hit that proportion instead of inheriting the sample&rsquo;s skew.
          </p>
        </div>
      )}
    </div>
  )
}

// -- preview + estimate -------------------------------------------------------

function PreviewTable({
  rows,
  seed,
}: {
  rows: Array<Record<string, unknown>>
  seed: number
}) {
  const reduced = useReducedMotion()
  const columns = rows.length > 0 ? Object.keys(rows[0]) : []
  return (
    <Card>
      <CardHeader>
        <CardTitle>Preview</CardTitle>
        <CardDescription>
          {rows.length} rows, seed <span className="font-mono">{seed}</span>. This is what the full
          run will look like — nothing has been saved yet.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <motion.div
          initial={reduced ? false : { opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: reduced ? 0 : 0.24 }}
          className="overflow-x-auto rounded-md border border-border"
        >
          <table className="w-full text-sm">
            <thead className="bg-muted/40">
              <tr>
                {columns.map((column) => (
                  <th key={column} className="px-3 py-2 text-left font-medium">
                    {column}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, index) => (
                <tr key={index} className="border-t border-border">
                  {columns.map((column) => (
                    <td key={column} className="px-3 py-1.5">
                      {String(row[column] ?? '')}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </motion.div>
      </CardContent>
    </Card>
  )
}

function EstimatePanel({
  estimate,
  onConfirm,
  running,
  canRun,
}: {
  estimate: { estimated_provider_calls: number; uses_provider: boolean; rows: number }
  onConfirm: () => void
  running: boolean
  canRun: boolean
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Ready to generate</CardTitle>
        <CardDescription>
          {fmtNumber(estimate.rows)} rows.{' '}
          {estimate.uses_provider
            ? `This will make about ${fmtNumber(estimate.estimated_provider_calls)} Provider call(s).`
            : 'This Mode needs no Provider calls at all.'}
        </CardDescription>
      </CardHeader>
      <CardFooter>
        <Button onClick={onConfirm} disabled={!canRun || running}>
          {running ? 'Running…' : 'Generate the full dataset'}
        </Button>
      </CardFooter>
    </Card>
  )
}
