import type { ReactNode } from 'react'

import { motion, useReducedMotion } from 'motion/react'

import { cn } from '@/lib/utils'

export interface StaggerGroupProps {
  children: ReactNode
  className?: string
  /** Seconds between each child's entrance. */
  stagger?: number
  /** Seconds before the first child enters. */
  delay?: number
}

/**
 * Container that plays its `StaggerItem` children in one after another.
 * Under reduced motion every item appears instantly (no stagger, no travel).
 */
export function StaggerGroup({
  children,
  className,
  stagger = 0.06,
  delay = 0.05,
}: StaggerGroupProps) {
  const reduced = useReducedMotion()
  return (
    <motion.div
      className={className}
      initial="hidden"
      animate="visible"
      variants={{
        hidden: {},
        visible: {
          transition: reduced
            ? { staggerChildren: 0 }
            : { staggerChildren: stagger, delayChildren: delay },
        },
      }}
    >
      {children}
    </motion.div>
  )
}

export interface StaggerItemProps {
  children: ReactNode
  className?: string
}

/** One entry inside a `StaggerGroup`; instant under reduced motion. */
export function StaggerItem({ children, className }: StaggerItemProps) {
  const reduced = useReducedMotion()
  return (
    <motion.div
      className={cn(className)}
      variants={
        reduced
          ? { hidden: { opacity: 0 }, visible: { opacity: 1, transition: { duration: 0 } } }
          : {
              hidden: { opacity: 0, y: 10 },
              visible: { opacity: 1, y: 0, transition: { duration: 0.3, ease: 'easeOut' } },
            }
      }
    >
      {children}
    </motion.div>
  )
}
