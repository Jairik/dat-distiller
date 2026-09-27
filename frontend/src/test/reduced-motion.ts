/**
 * Shared toggle for the `prefers-reduced-motion` media query stub installed
 * in `setup.ts`. Motion primitives read the real hook, so flipping this flag
 * exercises their reduced-motion branches end to end.
 */
export const matchMediaState = { reduce: false }

export function setPrefersReducedMotion(enabled: boolean): void {
  matchMediaState.reduce = enabled
}
