import { useEffect, useRef, useState } from 'react'

import { animate, useReducedMotion } from 'motion/react'

import { cn } from '@/lib/utils'

export interface AnimatedNumberProps {
  /** The number to land on. Changes animate from the value on screen. */
  value: number
  /** Seconds a transition takes (normal motion only). */
  duration?: number
  /** How to render the number while it counts. */
  format?: (value: number) => string
  className?: string
}

function defaultFormat(value: number): string {
  return Math.round(value).toLocaleString('en-US')
}

/**
 * Number that counts up to `value` when it mounts (and animates between
 * values afterwards). Under `prefers-reduced-motion: reduce` it simply shows
 * the final value — no counting at all.
 */
export function AnimatedNumber({
  value,
  duration = 0.6,
  format = defaultFormat,
  className,
}: AnimatedNumberProps) {
  const reduced = useReducedMotion() ?? false
  const startRef = useRef(reduced ? value : 0)
  const [shown, setShown] = useState(startRef.current)

  useEffect(() => {
    if (reduced) {
      startRef.current = value
      setShown(value)
      return
    }
    const controls = animate(startRef.current, value, {
      duration,
      ease: 'easeOut',
      onUpdate: (latest) => {
        startRef.current = latest
        setShown(latest)
      },
      onComplete: () => {
        startRef.current = value
      },
    })
    return () => controls.stop()
  }, [value, duration, reduced])

  return <span className={cn('tabular-nums', className)}>{format(shown)}</span>
}
