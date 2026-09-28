/**
 * A copyable command shown as code, with a copy affordance that degrades to a
 * manual copy when the Clipboard API is unavailable (see `lib/clipboard`).
 */

import { useEffect, useState } from 'react'
import { Button } from '@/components/ui/button'
import { copyText } from '@/lib/clipboard'
import { cn } from '@/lib/utils'

export interface CopyButtonProps {
  text: string
  /** Names the copy button for screen readers, e.g. "Codex install command". */
  label: string
  className?: string
}

export function CopyButton({ text, label, className }: CopyButtonProps) {
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    if (!copied) return
    const timer = setTimeout(() => setCopied(false), 1600)
    return () => clearTimeout(timer)
  }, [copied])

  return (
    <div className={cn('flex min-w-0 items-center gap-2', className)}>
      <code
        title={`Select and copy: ${text}`}
        className="min-w-0 flex-1 truncate rounded-md border border-border bg-muted/60 px-2 py-1 font-mono text-xs text-foreground"
      >
        {text}
      </code>
      <Button
        type="button"
        size="xs"
        variant="outline"
        aria-label={`Copy ${label}`}
        onClick={() => {
          void copyText(text).then(setCopied)
        }}
      >
        {copied ? 'Copied' : 'Copy'}
      </Button>
    </div>
  )
}
