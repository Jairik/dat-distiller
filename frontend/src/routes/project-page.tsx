/**
 * One Project: header + section tabs (Dataset · Generate · Label · Train).
 *
 * Child routes read this page's data through `useOutletContext`.
 */

import { Link, Outlet, useLocation, useOutletContext, useParams } from 'react-router-dom'
import { NavLink } from 'react-router-dom'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useProject, useVersions, type DatasetVersion, type Project } from '@/lib/projects'
import { fmtNumber } from '@/lib/format'

export interface ProjectContext {
  project: Project
  versions: DatasetVersion[]
  reloadVersions: () => void
}

export function useProjectContext(): ProjectContext {
  return useOutletContext<ProjectContext>()
}

const SECTIONS = [
  { key: 'dataset', label: 'Dataset' },
  { key: 'generate', label: 'Generate' },
  { key: 'label', label: 'Label' },
  { key: 'train', label: 'Train' },
] as const

export function ProjectPage() {
  const { projectId } = useParams()
  const location = useLocation()
  const project = useProject(projectId)
  const versions = useVersions(projectId)

  if (project.isPending) return <p className="text-sm text-muted-foreground">Loading…</p>
  if (project.isError) {
    return (
      <div className="flex flex-col items-start gap-3">
        <p className="text-sm text-destructive">
          {(project.error as Error & { status?: number }).status === 404
            ? 'Project not found.'
            : `Could not load project: ${(project.error as Error).message}`}
        </p>
        <Link to="/" className="text-sm text-primary underline">
          Back to Projects
        </Link>
      </div>
    )
  }

  const latest = versions.data?.at(-1)
  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-wrap items-center gap-3">
        <h1 className="text-2xl font-semibold tracking-tight">{project.data.name}</h1>
        {latest && (
          <Badge variant="outline" className="text-muted-foreground">
            v{latest.number} · {fmtNumber(latest.row_count)} rows · {latest.origin}
          </Badge>
        )}
      </header>
      <Tabs value={location.pathname.split('/').pop()} className="w-fit">
        <TabsList>
          {SECTIONS.map((section) => (
            <TabsTrigger key={section.key} value={section.key} asChild>
              <NavLink to={`/projects/${projectId}/${section.key}`}>{section.label}</NavLink>
            </TabsTrigger>
          ))}
        </TabsList>
      </Tabs>
      <Outlet
        context={
          {
            project: project.data,
            versions: versions.data ?? [],
            reloadVersions: () => {
              versions.refetch()
              project.refetch()
            },
          } satisfies ProjectContext
        }
      />
    </div>
  )
}
