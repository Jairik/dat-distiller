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

// jsdom has no pointer capture, and the Radix Slider calls it on pointerdown
// (`hasPointerCapture` to test, `setPointerCapture` to claim the element).
// Without these the drag throws three uncaught exceptions: every test still
// *passes*, but Vitest exits 1, so a broken test command is indistinguishable
// from a real failure. A no-op is right — nothing here reads a capture.
for (const method of ['hasPointerCapture', 'releasePointerCapture', 'setPointerCapture']) {
  if (!(method in Element.prototype)) {
    Object.defineProperty(Element.prototype, method, {
      configurable: true,
      writable: true,
      value: () => false,
    })
  }
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
