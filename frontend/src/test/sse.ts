/**
 * A controllable `EventSource` double, shared by every test that follows a job.
 *
 * Two things this is careful about, both learned the hard way:
 *
 * 1. **A stream may not exist yet when you want to push to it.** A component
 *    opens its EventSource in an effect, which runs after paint — so
 *    `findByText('Labeling run')` can resolve before the connection exists, and
 *    emitting into `.at(-1)` then hits the *previous* component's stream or
 *    nothing at all. That is a flake that only shows up when the machine is busy.
 *    `emitJobEvent` therefore waits for the stream to exist.
 * 2. **Emitting outside `act()` warns.** The helpers wrap the emit so React
 *    settles before the assertions run.
 */

import { act, waitFor } from '@testing-library/react'
import { vi } from 'vitest'

export class FakeEventSource {
  static instances: FakeEventSource[] = []

  closed = false
  listeners = new Map<string, (event: MessageEvent) => void>()

  constructor(public url: string) {
    FakeEventSource.instances.push(this)
  }

  addEventListener(name: string, fn: EventListener) {
    this.listeners.set(name, fn as (event: MessageEvent) => void)
  }

  close() {
    this.closed = true
  }

  onerror: ((event: unknown) => void) | null = null

  emit(name: string, data: unknown) {
    this.listeners.get(name)?.({ data: JSON.stringify(data) } as MessageEvent)
  }
}

/** Install the double. Call from `beforeEach`. */
export function stubEventSource() {
  FakeEventSource.instances = []
  vi.stubGlobal('EventSource', FakeEventSource)
}

/** The live stream for a job, once one exists. */
export async function waitForEventSource(jobId: string) {
  const url = `/api/jobs/${jobId}/events`
  return waitFor(() => {
    const found = FakeEventSource.instances.filter((s) => s.url === url && !s.closed)
    if (found.length === 0) throw new Error(`no EventSource for ${url} yet`)
    return found.at(-1) as FakeEventSource
  })
}

/** Push one event to a job's stream, waiting for the stream first. */
export async function emitJobEvent(jobId: string, name: string, data: unknown) {
  const stream = await waitForEventSource(jobId)
  await act(async () => {
    stream.emit(name, data)
  })
}
