import type { ReactNode } from 'react'

import { motion, useReducedMotion } from 'motion/react'

export interface PageInProps {
  children: ReactNode
  className?: string
  /** Seconds to wait before entering (handy for sequencing). */
  delay?: number
}

/**
 * Page-level entrance: a short rise-and-fade. Under
 * `prefers-reduced-motion: reduce` it collapses to an opacity-only fade.
 */
export function PageIn({ children, className, delay = 0 }: PageInProps) {
  const reduced = useReducedMotion()
  return (
    <motion.div
      className={className}
      initial={reduced ? { opacity: 0 } : { opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={
        reduced
          ? { duration: 0.15, delay }
          : { duration: 0.35, delay, ease: [0.21, 0.65, 0.29, 0.99] }
      }
    >
      {children}
    </motion.div>
  )
}
