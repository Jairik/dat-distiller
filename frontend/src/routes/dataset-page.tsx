/**
 * Dataset section: the Version tree on the left (lineage, provenance
 * breakdown, upload), the selected Version's preview and downloads on the
 * right. Selection lives in the `?v=` query param so it is deep-linkable.
 */

import { useNavigate, useSearchParams } from 'react-router-dom'
import { ChecksPanel } from '@/components/checks/checks-panel'
import { DataPreview } from '@/components/dataset/data-preview'
import { ProvenanceLegend } from '@/components/dataset/provenance-bar'
import { FidelityPanel } from '@/components/fidelity/fidelity-panel'
import { UploadDrop } from '@/components/dataset/upload-drop'
import { VersionTree } from '@/components/dataset/version-tree'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { fmtNumber, fmtDateTime } from '@/lib/format'
import { downloadUrl, useVersions } from '@/lib/projects'
import { useProjectContext } from '@/routes/project-page'

export function DatasetPage() {
  const { project } = useProjectContext()
  const versions = useVersions(project.id)
  const navigate = useNavigate()
  const [params, setParams] = useSearchParams()
  const list = versions.data ?? []
  const selectedId = params.get('v') ?? list.at(-1)?.id ?? null
  const selected = list.find((version) => version.id === selectedId) ?? null

  const select = (id: string) => {
    const next = new URLSearchParams(params)
    next.set('v', id)
    setParams(next, { replace: true })
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[300px_minmax(0,1fr)]">
      <aside className="flex flex-col gap-4">
        <UploadDrop projectId={project.id} onUploaded={select} />
        <VersionTree versions={list} selectedId={selectedId} onSelect={select} />
        <ProvenanceLegend />
      </aside>
      <section className="flex min-w-0 flex-col gap-4">
        {selected ? (
          <>
            <header className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 className="text-lg font-semibold">
                  Version {selected.number}
                  <Badge variant="outline" className="ml-2 capitalize">
                    {selected.origin}
                  </Badge>
                </h2>
                <p className="text-sm text-muted-foreground">
                  {fmtNumber(selected.row_count)} rows · {selected.columns.length} columns ·{' '}
                  {fmtDateTime(selected.created_at)}
                  {selected.seed !== null && selected.seed !== undefined && (
                    <span> · seed {selected.seed}</span>
                  )}
                </p>
              </div>
              <div className="flex gap-2">
                <Button asChild size="sm" variant="outline">
                  <a href={downloadUrl(selected.id, 'csv')} download>
                    Download CSV
                  </a>
                </Button>
                <Button asChild size="sm" variant="outline">
                  <a href={downloadUrl(selected.id, 'parquet')} download>
                    Parquet
                  </a>
                </Button>
              </div>
            </header>
            <DataPreview key={selected.id} versionId={selected.id} />
            {selected.origin === 'generated' && <FidelityPanel versionId={selected.id} />}
            {/* the closing panel of this step: the Checks the upload and any
                fidelity run raised on this Version, gating the way to Generate */}
            <ChecksPanel
              subjectType="dataset_version"
              subjectId={selected.id}
              continueLabel="Go to Generate"
              onContinue={() => navigate(`/projects/${project.id}/generate?from=${selected.id}`)}
            />
          </>
        ) : (
          <p className="text-sm text-muted-foreground">Select or upload a Dataset Version.</p>
        )}
      </section>
    </div>
  )
}
