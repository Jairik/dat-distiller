/**
 * Training Runs for this Project, each with its Model Card.
 *
 * The leaderboard proper is #37's job; this is the durable list of what has been
 * trained, which is also where a Model Card is reachable from. A run that failed
 * or was cancelled is listed too, with the reason — a run that vanished on failure
 * would leave you wondering whether you had imagined it.
 */

import { useNavigate } from 'react-router-dom'
import { motion, useReducedMotion } from 'motion/react'
import { TriangleAlertIcon } from 'lucide-react'

import { CardButton } from '@/components/cards/card-viewer'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { fmtDateTime, fmtNumber } from '@/lib/format'
import { formatPrimary, useTrainingRuns, type TrainingRunSummary } from '@/lib/runs'
import { useProjectContext } from '@/routes/project-page'

export function TrainingRunsPanel() {
  const { project } = useProjectContext()
  const runs = useTrainingRuns(project.id)

  return (
    <Card data-testid="training-runs">
      <CardHeader>
        <CardTitle>Training Runs</CardTitle>
        <CardDescription>
          Every Training Run for this Project, newest first. A Model Card records what was
          fitted, how it scored, and what a person acknowledged along the way.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        {runs.isPending && <p className="text-sm text-muted-foreground">Reading the runs…</p>}
        {runs.isError && (
          <p role="alert" className="text-sm text-destructive">
            {(runs.error as Error).message}
          </p>
        )}
        {runs.data && runs.data.runs.length === 0 && (
          <p className="text-sm text-muted-foreground">
            {'No Training Runs yet. Train a Model and it will appear here.'}
          </p>
        )}
        {runs.data?.runs.map((run) => (
          <RunRow key={run.training_run_id} run={run} projectId={project.id} />
        ))}
      </CardContent>
    </Card>
  )
}

function RunRow({ run, projectId }: { run: TrainingRunSummary; projectId: string }) {
  const reduced = useReducedMotion()
  const navigate = useNavigate()
  const failed = run.status === 'failed' || run.status === 'cancelled'
  const busy = run.status === 'queued' || run.status === 'running'

  return (
    <motion.div
      initial={reduced ? false : { opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: reduced ? 0 : 0.2 }}
      data-testid={`training-run-${run.training_run_id}`}
      className={`flex flex-wrap items-start justify-between gap-3 rounded-lg border p-3 ${
        failed ? 'border-destructive/40 bg-destructive/5' : 'border-border'
      }`}
    >
      <div className="flex min-w-0 flex-col gap-1">
        <span className="flex flex-wrap items-center gap-2 text-sm font-medium">
          {run.target}
          <Badge variant="outline" className="text-xs">
            {run.task_type}
          </Badge>
          <Badge
            variant="outline"
            className={
              failed
                ? 'border-destructive/40 text-destructive'
                : run.status === 'completed'
                  ? 'border-emerald-400/40 text-emerald-400'
                  : 'text-muted-foreground'
            }
          >
            {run.status}
          </Badge>
        </span>

        {run.status === 'completed' ? (
          <span className="text-sm text-muted-foreground">
            {`${run.n_models} Model${run.n_models === 1 ? '' : 's'} · best: ${
              run.best_model ?? 'none ranked'
            } · ${run.primary_metric} ${formatPrimary(run.best_primary_value)}`}
          </span>
        ) : failed ? (
          <span className="flex items-start gap-1.5 text-sm text-destructive">
            <TriangleAlertIcon aria-hidden className="mt-0.5 size-3.5 shrink-0" />
            {run.error ?? (run.status === 'cancelled' ? 'cancelled' : 'failed')}
          </span>
        ) : (
          <span className="text-sm text-muted-foreground">{`still ${run.status}…`}</span>
        )}

        <span className="text-xs text-muted-foreground">
          {`seed ${run.seed} · ${fmtNumber(run.n_models)} model(s) · ${fmtDateTime(run.finished_at ?? run.created_at)}`}
        </span>

        {run.warnings.length > 0 && (
          <ul className="flex flex-col gap-0.5 text-xs text-amber-400">
            {run.warnings.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
        )}
      </div>

      <div className="flex gap-2">
        <CardButton
          doc={{
            kind: 'model',
            id: run.training_run_id,
            title: `${run.target} · ${run.task_type}`,
            subtitle: `seed ${run.seed}`,
            href: `/projects/${projectId}/train?run=${run.training_run_id}`,
          }}
        />
        {run.version_id && (
          <button
            type="button"
            className="text-xs text-muted-foreground underline hover:text-foreground"
            onClick={() => navigate(`/projects/${projectId}/dataset?v=${run.version_id}`)}
          >
            {busy ? 'The data' : 'See the data'}
          </button>
        )}
      </div>
    </motion.div>
  )
}
