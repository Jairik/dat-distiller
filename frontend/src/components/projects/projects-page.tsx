/**
 * Projects landing: a card grid, a create dialog with inline 409 conflict
 * handling, and a confirm dialog for the destructive delete.
 */

import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { ApiError } from '@/api/client'
import { StaggerGroup, StaggerItem } from '@/components/motion'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { useCreateProject, useDeleteProject, useProjects, isNameConflict, type Project } from '@/lib/projects'
import { timeAgo } from '@/lib/format'

export function ProjectsPage() {
  const navigate = useNavigate()
  const projects = useProjects()
  const create = useCreateProject()
  const remove = useDeleteProject()
  const [createOpen, setCreateOpen] = useState(false)
  const [name, setName] = useState('')
  const [confirmDelete, setConfirmDelete] = useState<Project | null>(null)

  async function submit() {
    const trimmed = name.trim()
    if (!trimmed) return
    try {
      const project = await create.mutateAsync(trimmed)
      setCreateOpen(false)
      setName('')
      navigate(`/projects/${project.id}/dataset`)
    } catch {
      // error surfaces inline below
    }
  }

  const createError = create.isError ? create.error : null

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Projects</h1>
          <p className="text-sm text-muted-foreground">
            Each project keeps its own Dataset Version tree.
          </p>
        </div>
        <Button onClick={() => setCreateOpen(true)}>New Project</Button>
      </div>

      {projects.isPending && <p className="text-sm text-muted-foreground">Loading projects…</p>}
      {projects.isError && (
        <p role="alert" className="text-sm text-destructive">
          Could not load projects: {(projects.error as Error).message}
        </p>
      )}
      {projects.data?.length === 0 && (
        <Card>
          <CardHeader>
            <CardTitle>No projects yet</CardTitle>
            <CardDescription>
              Create a project to start generating, labeling, or uploading data.
            </CardDescription>
          </CardHeader>
        </Card>
      )}

      <StaggerGroup className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {(projects.data ?? []).map((project) => (
          <StaggerItem key={project.id}>
            <Card className="group relative transition-colors hover:border-primary/40">
              <CardHeader>
                <CardTitle>
                  <Link to={`/projects/${project.id}/dataset`} className="after:absolute after:inset-0">
                    {project.name}
                  </Link>
                </CardTitle>
                <CardDescription>
                  {project.version_count} version{project.version_count === 1 ? '' : 's'} ·{' '}
                  {timeAgo(project.created_at)}
                </CardDescription>
              </CardHeader>
              <CardContent className="flex items-center justify-between">
                <Badge variant="outline">
                  {project.latest_version_id ? 'active' : 'empty'}
                </Badge>
                <Button
                  size="sm"
                  variant="ghost"
                  className="relative z-10 opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100"
                  onClick={() => setConfirmDelete(project)}
                >
                  Delete
                </Button>
              </CardContent>
            </Card>
          </StaggerItem>
        ))}
      </StaggerGroup>

      <Dialog open={createOpen} onOpenChange={setCreateOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>New Project</DialogTitle>
            <DialogDescription>Project names must be unique.</DialogDescription>
          </DialogHeader>
          <div className="grid gap-2">
            <Label htmlFor="project-name">Name</Label>
            <Input
              id="project-name"
              autoFocus
              value={name}
              onChange={(e) => setName(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && submit()}
            />
            {createError && (
              <p role="alert" className="text-sm text-destructive">
                {isNameConflict(createError)
                  ? 'A project with that name already exists.'
                  : (createError as ApiError).message}
              </p>
            )}
          </div>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setCreateOpen(false)}>
              Cancel
            </Button>
            <Button onClick={submit} disabled={create.isPending || !name.trim()}>
              {create.isPending ? 'Creating…' : 'Create'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={confirmDelete !== null} onOpenChange={(open) => !open && setConfirmDelete(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete “{confirmDelete?.name}”?</DialogTitle>
            <DialogDescription>
              This removes the project and all of its Dataset Versions. There is no undo.
            </DialogDescription>
          </DialogHeader>
          {/* A delete that fails must say so. `mutateAsync` rejects, and an
              unhandled rejection here left the dialog open with nothing said —
              a destructive action that failed looked exactly like a button that
              did nothing, and the Project is still there, so the answer is to
              try again. Same shape as the create dialog's error above. */}
          {remove.isError && (
            <p role="alert" className="text-sm text-destructive">
              {(remove.error as ApiError).message}
            </p>
          )}
          <DialogFooter>
            <Button variant="ghost" onClick={() => setConfirmDelete(null)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              disabled={remove.isPending}
              onClick={async () => {
                if (!confirmDelete) return
                try {
                  await remove.mutateAsync(confirmDelete.id)
                  setConfirmDelete(null)
                } catch {
                  // The dialog stays open and the error is shown above it, so the
                  // Project is still there to try again.
                }
              }}
            >
              {remove.isPending ? 'Deleting…' : 'Delete project'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
