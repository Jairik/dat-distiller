import type { ReactNode } from 'react'

import { motion, useReducedMotion } from 'motion/react'

export interface HoverPressProps {
  children: ReactNode
  className?: string
  /** Scale while hovered (normal motion only). */
  hoverScale?: number
  /** Scale while pressed (normal motion only). */
  pressScale?: number
}

/**
 * Micro-interaction wrapper: gently grows on hover, presses in on click.
 * Under `prefers-reduced-motion: reduce` the scale changes are dropped
 * entirely — the element just behaves like a plain box.
 */
export function HoverPress({
  children,
  className,
  hoverScale = 1.02,
  pressScale = 0.97,
}: HoverPressProps) {
  const reduced = useReducedMotion()
  return (
    <motion.div
      className={className}
      whileHover={reduced ? undefined : { scale: hoverScale }}
      whileTap={reduced ? undefined : { scale: pressScale }}
      transition={{ type: 'spring', stiffness: 400, damping: 25 }}
    >
      {children}
    </motion.div>
  )
}
