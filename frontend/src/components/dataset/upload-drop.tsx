/**
 * Drag-and-drop upload (CSV / Parquet / JSONL): the first dropped or picked
 * file becomes a new root Dataset Version; errors show inline.
 */

import { useRef, useState } from 'react'
import { useUpload } from '@/lib/projects'
import { cn } from '@/lib/utils'

export function UploadDrop({
  projectId,
  onUploaded,
}: {
  projectId: string
  onUploaded?: (versionId: string) => void
}) {
  const upload = useUpload(projectId)
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragOver, setDragOver] = useState(false)

  const handleFiles = (files: FileList | null) => {
    const file = files?.[0]
    if (!file) return
    upload.mutate(file, {
      onSuccess: (version) => onUploaded?.(version.id),
    })
  }

  return (
    <div className="flex flex-col gap-1.5">
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault()
          setDragOver(true)
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragOver(false)
          handleFiles(e.dataTransfer.files)
        }}
        className={cn(
          'flex flex-col items-center justify-center gap-1 rounded-lg border border-dashed border-border px-3 py-5 text-center text-sm text-muted-foreground transition-colors hover:border-primary/50 hover:text-foreground',
          dragOver && 'border-primary bg-primary/5 text-foreground',
        )}
      >
        {upload.isPending ? (
          <span>Uploading…</span>
        ) : (
          <>
            <span className="font-medium text-foreground">Upload data</span>
            <span>drop a file or click — .csv, .parquet, .jsonl</span>
          </>
        )}
      </button>
      <input
        ref={inputRef}
        type="file"
        accept=".csv,.parquet,.jsonl"
        className="hidden"
        aria-label="Upload data file"
        onChange={(e) => handleFiles(e.target.files)}
      />
      {upload.isError && (
        <p role="alert" className="text-sm text-destructive">
          {(upload.error as Error).message}
        </p>
      )}
    </div>
  )
}
