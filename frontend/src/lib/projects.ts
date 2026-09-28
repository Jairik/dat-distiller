/**
 * Projects + Dataset Versions: types and TanStack Query hooks.
 *
 * Versions arrive as a FLAT list ordered by creation; `parent_id` (null for
 * roots) is what the version-tree view builds on.
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ApiError, apiDelete, apiGet, apiPost } from '@/api/client'

export interface Project {
  id: string
  name: string
  created_at: string
  version_count: number
  latest_version_id: string | null
}

export interface VersionColumn {
  name: string
  kind: string
  [key: string]: unknown
}

export interface DatasetVersion {
  id: string
  project_id: string
  parent_id: string | null
  number: number
  origin: string
  row_count: number
  columns: VersionColumn[]
  provenance_summary: Record<string, number>
  seed: number | null
  meta: Record<string, unknown>
  created_at: string
}

export interface VersionPreview {
  version_id: string
  columns: Array<Record<string, unknown>>
  page: number
  page_size: number
  total_rows: number
  rows: unknown[][]
}

export function useProjects() {
  return useQuery({
    queryKey: ['projects'],
    queryFn: () => apiGet<Project[]>('/projects'),
  })
}

export function useProject(id: string | undefined) {
  return useQuery({
    queryKey: ['project', id],
    queryFn: () => apiGet<Project>(`/projects/${id}`),
    enabled: Boolean(id),
  })
}

export function useVersions(id: string | undefined) {
  return useQuery({
    queryKey: ['versions', id],
    queryFn: () => apiGet<DatasetVersion[]>(`/projects/${id}/dataset_versions`),
    enabled: Boolean(id),
  })
}

export function usePreview(id: string | undefined, page = 1, pageSize = 25) {
  return useQuery({
    queryKey: ['preview', id, page, pageSize],
    queryFn: () =>
      apiGet<VersionPreview>(`/dataset-versions/${id}/preview?page=${page}&page_size=${pageSize}`),
    enabled: Boolean(id),
  })
}

export function useCreateProject() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (name: string) => apiPost<Project>('/projects', { name }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['projects'] }),
  })
}

export function useDeleteProject() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => apiDelete(`/projects/${id}`),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['projects'] }),
  })
}

/** True when a mutation error is FastAPI's "name already taken" conflict. */
export function isNameConflict(error: unknown): error is ApiError {
  return error instanceof ApiError && error.status === 409
}
