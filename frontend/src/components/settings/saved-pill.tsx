/**
 * Confirmation feedback for a saved change: an animated pill that fades in,
 * sits for ~2s, then fades out. Motion stays inside the pill (and honours
 * `prefers-reduced-motion`), so a save never re-animates the whole page.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'

export interface SavedFlash {
  id: number
  message: string
}

export type SavedNotifier = (message?: string) => void

/** How long the confirmation stays up before fading away. */
const VISIBLE_MS = 2000

export function useSavedFlash() {
  const [flash, setFlash] = useState<SavedFlash | null>(null)
  const seq = useRef(0)

  const saved = useCallback<SavedNotifier>((message = 'Saved') => {
    seq.current += 1
    setFlash({ id: seq.current, message })
  }, [])

  useEffect(() => {
    if (!flash) return
    const timer = setTimeout(() => setFlash(null), VISIBLE_MS)
    return () => clearTimeout(timer)
  }, [flash])

  return { flash, saved }
}

export function SavedPill({ flash }: { flash: SavedFlash | null }) {
  const reduced = useReducedMotion()
  return (
    <div aria-live="polite" className="pointer-events-none fixed inset-x-0 bottom-6 z-50 flex justify-center px-6">
      <AnimatePresence mode="wait">
        {flash && (
          <motion.span
            key={flash.id}
            role="status"
            initial={reduced ? { opacity: 0 } : { opacity: 0, y: 10, scale: 0.96 }}
            animate={reduced ? { opacity: 1 } : { opacity: 1, y: 0, scale: 1 }}
            exit={reduced ? { opacity: 0 } : { opacity: 0, y: -8, scale: 0.98 }}
            transition={{ duration: reduced ? 0.12 : 0.24, ease: [0.21, 0.65, 0.29, 0.99] }}
            className="pointer-events-auto inline-flex items-center gap-1.5 rounded-full border border-primary/40 bg-primary/10 px-3 py-1.5 text-xs font-medium text-primary shadow-sm backdrop-blur"
          >
            <svg viewBox="0 0 16 16" aria-hidden="true" className="size-3.5 fill-none stroke-current stroke-2">
              <path d="M2.5 8.5l3.5 3.5 7.5-8" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            {flash.message}
          </motion.span>
        )}
      </AnimatePresence>
    </div>
  )
}
