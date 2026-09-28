import '@testing-library/jest-dom/vitest'

import { matchMediaState } from './reduced-motion'

// jsdom has no ResizeObserver, and Radix primitives construct one at mount
// (the Slider, and anything else that measures itself) — without this the
// component throws and takes the whole render down with it. A no-op is enough:
// these tests assert behaviour, not layout, and no code here reads a size.
if (!('ResizeObserver' in globalThis)) {
  class NoopResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
  Object.defineProperty(globalThis, 'ResizeObserver', {
    writable: true,
    configurable: true,
    value: NoopResizeObserver,
  })
}

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
