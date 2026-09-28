/**
 * Playwright configuration for the end-to-end suite.
 *
 * One command runs everything: `npm run e2e`. It builds the frontend, then
 * starts the real backend against a throwaway data directory with
 * `FakeProvider` and `FakeJev`, so the suite needs no network and no API keys —
 * which is the point. A test that could fail because a Provider had a bad day is
 * not a test of this app.
 *
 * The server is the *same* `dat-distiller serve` a user runs, serving the
 * *same* built `frontend/dist`. Nothing is mocked at the HTTP boundary, so what
 * the suite exercises is the wiring, not a fixture of it.
 */

import { defineConfig, devices } from '@playwright/test'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const here = path.dirname(fileURLToPath(import.meta.url))
const repoRoot = path.resolve(here, '..')
const backend = path.join(repoRoot, 'backend')

/** A free-ish port, distinct from the dev server, so both can run at once. */
const PORT = Number(process.env.DAT_DISTILLER_E2E_PORT ?? 8757)
const baseURL = `http://127.0.0.1:${PORT}`

/** A throwaway data directory, so a run never touches a real Project's data. */
const dataDir = path.join(repoRoot, '.e2e-data')

export default defineConfig({
  testDir: './e2e',
  // the suite walks a whole app; running the specs in parallel would have them
  // sharing one server and one SQLite file
  fullyParallel: false,
  workers: 1,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  // a job-backed flow is slower than a click, and a retry should not paper over
  // a genuinely broken step
  timeout: 120_000,
  expect: { timeout: 15_000 },
  reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : [['list']],
  use: {
    baseURL,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'off',
    // the app is a local, single-user tool; nothing here needs a real download
    // directory, and Playwright's own is the safest place for one
    acceptDownloads: true,
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: {
    // Three things, in order, and all of them matter:
    //  - `uv sync --extra …` so the server has scikit-learn and LightGBM. Without
    //    them the Train step has *no runnable Models* and the suite would have
    //    nothing to train — the extras are optional, so a bare sync omits them.
    //  - `vite build`, because the backend serves frontend/dist and a server
    //    started against a missing build serves a blank page, not an error.
    //  - `dat-distiller serve`: the same command a user runs.
    // The build runs through --prefix because this config lives in frontend/
    // while the command's cwd is the repo root.
    command:
      `uv sync --project ${JSON.stringify(backend)} --extra sklearn --extra lightgbm && ` +
      `npm --prefix ${JSON.stringify(here)} run build && ` +
      `uv run --project ${JSON.stringify(backend)} dat-distiller serve --port ${PORT}`,
    cwd: repoRoot,
    url: `${baseURL}/api/health`,
    reuseExistingServer: !process.env.CI,
    timeout: 240_000,
    stdout: 'pipe',
    stderr: 'pipe',
    env: {
      // no network and no keys: a Provider or a Jev that reaches the network is
      // a bug in the fakes, not a slow test
      DAT_DISTILLER_FAKE_PROVIDERS: '1',
      DAT_DISTILLER_FAKE_JEV: '1',
      DAT_DISTILLER_DATA_DIR: dataDir,
      DAT_DISTILLER_CONFIG_DIR: path.join(dataDir, 'config'),
      // a fixed port for the extras endpoint, so the UI can show which Models
      // this run could use without guessing
      PYTHONUNBUFFERED: '1',
    },
  },
})
