import '@testing-library/jest-dom/vitest'

import { matchMediaState } from './reduced-motion'

// jsdom has no usable matchMedia; motion's useReducedMotion needs one.
// `matchMediaState.reduce` steers it — see src/test/reduced-motion.ts.
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  configurable: true,
  value: (query: string): MediaQueryList => ({
    matches: matchMediaState.reduce && query.includes('prefers-reduced-motion'),
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  }),
})
