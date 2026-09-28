/**
 * Paginated data preview for one Dataset Version, with column type badges.
 * The parent passes `key={versionId}` so switching versions resets the page.
 */

import { useState } from 'react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { fmtNumber } from '@/lib/format'
import { usePreview } from '@/lib/projects'

function renderCell(value: unknown): React.ReactNode {
  if (value === null || value === undefined) {
    return <span className="text-muted-foreground/60">—</span>
  }
  if (typeof value === 'number') return <span className="tabular-nums">{fmtNumber(value)}</span>
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

export function DataPreview({ versionId }: { versionId: string }) {
  const [page, setPage] = useState(1)
  const preview = usePreview(versionId, page)

  if (preview.isPending) return <p className="text-sm text-muted-foreground">Loading rows…</p>
  if (preview.isError)
    return (
      <p role="alert" className="text-sm text-destructive">
        {(preview.error as Error).message}
      </p>
    )

  const data = preview.data
  const from = (data.page - 1) * data.page_size + 1
  const to = Math.min(data.page * data.page_size, data.total_rows)
  const pages = Math.max(Math.ceil(data.total_rows / data.page_size), 1)

  return (
    <div className="flex flex-col gap-2">
      <div className="overflow-x-auto rounded-md border border-border">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border bg-muted/40 text-left">
              {data.columns.map((column) => (
                <th key={String(column.name)} className="px-3 py-2 font-medium">
                  <span className="flex items-center gap-2 whitespace-nowrap">
                    {String(column.name)}
                    <Badge variant="outline" className="text-[10px] font-normal">
                      {String(column.kind ?? '?')}
                    </Badge>
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.rows.map((row, rowIndex) => (
              <tr key={rowIndex} className="border-b border-border/50 last:border-0">
                {row.map((cell, cellIndex) => (
                  <td key={cellIndex} className="max-w-64 truncate px-3 py-1.5">
                    {renderCell(cell)}
                  </td>
                ))}
              </tr>
            ))}
            {data.rows.length === 0 && (
              <tr>
                <td
                  colSpan={Math.max(data.columns.length, 1)}
                  className="px-3 py-6 text-center text-muted-foreground"
                >
                  This version has no rows.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      <div className="flex items-center justify-between text-sm text-muted-foreground">
        <span>
          {data.total_rows === 0
            ? '0 rows'
            : `rows ${fmtNumber(from)}–${fmtNumber(to)} of ${fmtNumber(data.total_rows)}`}
        </span>
        <span className="flex items-center gap-2">
          <Button size="sm" variant="ghost" disabled={data.page <= 1} onClick={() => setPage(page - 1)}>
            Previous
          </Button>
          <span>
            {data.page} / {pages}
          </span>
          <Button size="sm" variant="ghost" disabled={data.page >= pages} onClick={() => setPage(page + 1)}>
            Next
          </Button>
        </span>
      </div>
    </div>
  )
}
