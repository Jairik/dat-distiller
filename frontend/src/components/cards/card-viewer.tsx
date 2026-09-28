/**
 * Viewing a Card: rendered Markdown in a drawer, downloadable as `.md`.
 *
 * A Card is a *record*, not a decoration, so the drawer is plain and legible:
 * tables render as tables, monospaced text stays monospaced, and the raw
 * Markdown is one keystroke away for anyone who wants to read what was actually
 * written rather than our rendering of it.
 *
 * The fetch happens on open, not on mount — the history panel can hold dozens of
 * versions and firing one request per row to build buttons nobody clicked would
 * be rude.
 */

import { useEffect, useState, type ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, useReducedMotion } from 'motion/react'
import { DownloadIcon, FileTextIcon } from 'lucide-react'

import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'

export type CardKind = 'dataset' | 'model'

export interface CardDoc {
  kind: CardKind
  /** Dataset Version id, or Training Run id. */
  id: string
  title: string
  /** e.g. "v3 · 500 rows". */
  subtitle?: string
  /** Where "Go to it" navigates to. */
  href?: string
}

export function cardUrl(doc: CardDoc, format: 'markdown' | 'json' = 'markdown'): string {
  const base =
    doc.kind === 'dataset'
      ? `/api/dataset-versions/${doc.id}/card`
      : `/api/train/runs/${doc.id}/card`
  return `${base}?format=${format}`
}

/** The button that opens a Card, and the drawer it opens. */
export function CardButton({ doc }: { doc: CardDoc }) {
  const [open, setOpen] = useState(false)
  return (
    <>
      <Button size="sm" variant="outline" onClick={() => setOpen(true)}>
        <FileTextIcon aria-hidden />
        {doc.kind === 'dataset' ? 'Dataset Card' : 'Model Card'}
      </Button>
      <CardDrawer doc={doc} open={open} onOpenChange={setOpen} />
    </>
  )
}

export function CardDrawer({
  doc,
  open,
  onOpenChange,
}: {
  doc: CardDoc
  open: boolean
  onOpenChange: (open: boolean) => void
}) {
  const [format, setFormat] = useState<'rendered' | 'markdown'>('rendered')
  const [state, setState] = useState<{ markdown: string; error: string | null } | null>(null)
  const [loading, setLoading] = useState(false)
  const reduced = useReducedMotion()
  const navigate = useNavigate()
  const url = cardUrl(doc)

  useEffect(() => {
    if (!open) {
      setState(null)
      return
    }
    let cancelled = false
    setLoading(true)
    fetch(url)
      .then(async (response) =>
        response.ok
          ? { markdown: await response.text(), error: null }
          : { markdown: '', error: `Could not read this Card (${response.status}).` },
      )
      .then((next) => {
        if (!cancelled) setState(next)
      })
      .catch(() => {
        if (!cancelled) setState({ markdown: '', error: 'Could not read this Card.' })
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [open, url])

  const label = doc.kind === 'dataset' ? 'Dataset Card' : 'Model Card'

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="flex max-h-[85vh] flex-col overflow-hidden sm:max-w-3xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <FileTextIcon aria-hidden className="size-4" />
            {label}
          </DialogTitle>
          <DialogDescription>
            {doc.title}
            {doc.subtitle ? ` · ${doc.subtitle}` : ''}
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-wrap items-center justify-between gap-2">
          <Tabs value={format} onValueChange={(value) => setFormat(value as 'rendered' | 'markdown')}>
            <TabsList>
              <TabsTrigger value="rendered">Rendered</TabsTrigger>
              <TabsTrigger value="markdown">Markdown</TabsTrigger>
            </TabsList>
          </Tabs>
          <div className="flex gap-2">
            <Button size="sm" variant="outline" asChild>
              <a href={url} download={`${doc.kind}-card.md`}>
                <DownloadIcon aria-hidden />
                Download .md
              </a>
            </Button>
            {doc.href && (
              <Button size="sm" variant="ghost" onClick={() => navigate(doc.href as string)}>
                Go to it
              </Button>
            )}
          </div>
        </div>

        <div
          data-testid="card-body"
          className="min-h-0 flex-1 overflow-y-auto rounded-md border border-border bg-background/60 p-4"
        >
          {loading && <p className="text-sm text-muted-foreground">Reading the Card…</p>}
          {state?.error && (
            <p role="alert" className="text-sm text-destructive">
              {state.error}
            </p>
          )}
          {state?.markdown && format === 'rendered' && (
            <motion.div
              initial={reduced ? false : { opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: reduced ? 0 : 0.24 }}
            >
              <Markdownish text={state.markdown} />
            </motion.div>
          )}
          {state?.markdown && format === 'markdown' && (
            <pre data-testid="card-raw" className="whitespace-pre-wrap text-xs leading-relaxed">
              {state.markdown}
            </pre>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}

/**
 * Enough Markdown for a Card: headings, bold, inline code, links, lists, tables
 * and the italic-aside lines. Deliberately tiny — Cards are generated by our own
 * code, so this only has to handle what we emit, and a full Markdown library
 * would be a large dependency for a fixed job.
 */
export function Markdownish({ text }: { text: string }): ReactNode {
  const lines = text.split('\n')
  const blocks: ReactNode[] = []
  let index = 0
  let key = 0

  while (index < lines.length) {
    const line = lines[index]

    if (!line.trim()) {
      index += 1
      continue
    }

    if (line.startsWith('### ')) {
      blocks.push(
        <h4 key={key++} className="mt-3 text-sm font-semibold">
          {inline(line.slice(4))}
        </h4>,
      )
      index += 1
      continue
    }
    if (line.startsWith('## ')) {
      blocks.push(
        <h3 key={key++} className="mt-5 text-base font-semibold first:mt-0">
          {inline(line.slice(3))}
        </h3>,
      )
      index += 1
      continue
    }
    if (line.startsWith('# ')) {
      blocks.push(
        <h2 key={key++} className="text-lg font-semibold">
          {inline(line.slice(2))}
        </h2>,
      )
      index += 1
      continue
    }
    if (line.startsWith('- ')) {
      const items: string[] = []
      while (index < lines.length && lines[index].startsWith('- ')) {
        items.push(lines[index].slice(2))
        index += 1
      }
      blocks.push(
        <ul key={key++} className="ml-5 list-disc text-sm">
          {items.map((item, i) => (
            <li key={i}>{inline(item)}</li>
          ))}
        </ul>,
      )
      continue
    }
    if (line.startsWith('|')) {
      const rows: string[][] = []
      while (index < lines.length && lines[index].startsWith('|')) {
        rows.push(
          lines[index]
            .split('|')
            .slice(1, -1)
            .map((cell) => cell.trim()),
        )
        index += 1
      }
      const [head, divider, ...body] = rows
      // a table's second line is the |---|---| separator, not data
      const isDivider = (row: string[] | undefined) =>
        Boolean(row) && row!.every((cell) => /^:?-{2,}:?$/.test(cell))
      const data = isDivider(divider) ? body : rows.slice(1)
      blocks.push(
        <div key={key++} className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr>
                {(head ?? []).map((cell, i) => (
                  <th key={i} className="border-b border-border px-2 py-1 text-left font-medium">
                    {inline(cell)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.map((row, i) => (
                <tr key={i} className="align-top">
                  {row.map((cell, j) => (
                    <td key={j} className="px-2 py-1">
                      {inline(cell)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>,
      )
      continue
    }
    if (/^_.*_$/.test(line.trim())) {
      blocks.push(
        <p key={key++} className="text-xs italic text-muted-foreground">
          {inline(line.trim().slice(1, -1))}
        </p>,
      )
      index += 1
      continue
    }
    blocks.push(
      <p key={key++} className="text-sm">
        {inline(line)}
      </p>,
    )
    index += 1
  }

  return <div className="flex flex-col gap-2">{blocks}</div>
}

/** `**bold**`, `` `code` `` and `[text](url)` inside one line. */
export function inline(text: string): ReactNode {
  const parts: ReactNode[] = []
  const pattern = /(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))/g
  let cursor = 0
  let match: RegExpExecArray | null
  let key = 0
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > cursor) parts.push(text.slice(cursor, match.index))
    const token = match[0]
    if (token.startsWith('**')) {
      parts.push(<strong key={key++}>{token.slice(2, -2)}</strong>)
    } else if (token.startsWith('`')) {
      parts.push(
        <code key={key++} className="rounded bg-muted px-1 py-0.5 font-mono text-xs">
          {token.slice(1, -1)}
        </code>,
      )
    } else {
      const link = /^\[([^\]]+)\]\(([^)]+)\)$/.exec(token)
      if (link) {
        parts.push(
          <a key={key++} href={link[2]} className="text-primary underline">
            {link[1]}
          </a>,
        )
      }
    }
    cursor = match.index + token.length
  }
  if (cursor < text.length) parts.push(text.slice(cursor))
  return parts.length ? parts : text
}
