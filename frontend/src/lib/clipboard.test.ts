/**
 * `copyText` has to work in two worlds: a normal browser with the async
 * Clipboard API, and a plain-http LAN visit where `navigator.clipboard` is
 * missing or refused. It reports success either way and never throws, so the
 * caller can leave the command on screen for a manual copy.
 */

import { afterEach, describe, expect, it, vi } from 'vitest'

import { copyText } from '@/lib/clipboard'

function setClipboard(value: unknown) {
  Object.defineProperty(navigator, 'clipboard', { value, configurable: true })
}

afterEach(() => {
  setClipboard(undefined)
})

describe('copyText', () => {
  it('uses the async Clipboard API when it is there', async () => {
    const writeText = vi.fn(async () => undefined)
    setClipboard({ writeText })
    expect(await copyText('uv sync --extra torch')).toBe(true)
    expect(writeText).toHaveBeenCalledWith('uv sync --extra torch')
  })

  it('falls back to execCommand when there is no Clipboard API', async () => {
    setClipboard(undefined)
    const exec = vi.fn(() => true)
    document.execCommand = exec as unknown as typeof document.execCommand
    expect(await copyText('npm i -g @openai/codex')).toBe(true)
    expect(exec).toHaveBeenCalledWith('copy')
    // the scratch textarea is cleaned up, not left in the document
    expect(document.querySelectorAll('textarea')).toHaveLength(0)
  })

  it('falls back when the Clipboard API is refused', async () => {
    setClipboard({
      writeText: vi.fn(async () => {
        throw new Error('not focused')
      }),
    })
    const exec = vi.fn(() => true)
    document.execCommand = exec as unknown as typeof document.execCommand
    expect(await copyText('anything')).toBe(true)
    expect(exec).toHaveBeenCalledWith('copy')
  })

  it('reports failure instead of throwing, so the value stays on screen', async () => {
    setClipboard(undefined)
    document.execCommand = vi.fn(() => {
      throw new Error('blocked')
    }) as unknown as typeof document.execCommand
    await expect(copyText('anything')).resolves.toBe(false)
  })
})
