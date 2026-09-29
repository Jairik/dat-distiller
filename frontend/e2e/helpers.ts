/**
 * Shared helpers for the end-to-end suite.
 *
 * Two rules, because an end-to-end suite that breaks for the wrong reason is
 * worse than no suite:
 *
 * 1. **Wait for the thing, never for a duration.** `expect(...).toBeVisible()`
 *    retries; `waitForTimeout(2000)` is a coin flip. Every wait here is on
 *    something a user can see.
 * 2. **Fail loudly, and say what to do.** A helper that times out throws with the
 *    page's visible text, so a failure names the step that broke rather than
 *    "element not found".
 */

import { expect, type Locator, type Page } from '@playwright/test'

/** Create a Project and land on its Dataset step. Returns its name. */
export async function createProject(page: Page, name: string): Promise<string> {
  await page.goto('/')
  await page.getByRole('button', { name: 'New Project' }).click()
  const dialog = page.getByRole('dialog')
  await dialog.getByRole('textbox').fill(name)
  await dialog.getByRole('button', { name: 'Create' }).click()
  await expect(page.getByText(name, { exact: true }).first()).toBeVisible()
  return name
}

/**
 * Set the Column Specs to exactly these columns.
 *
 * The Spec table starts with one placeholder column, so "add three" leaves four
 * and the leftover one is an empty `number` with no bounds — which fails much
 * later, at run time, for a reason that has nothing to do with the test. So the
 * list is reconciled to the target rather than appended to.
 */
export async function setColumnSpecs(
  page: Page,
  columns: Array<{ name: string; type: string; value: string }>,
): Promise<void> {
  await page.getByRole('radio', { name: 'Define the columns' }).check()

  const removeLast = async () => {
    const removes = page.getByRole('button', { name: /^Remove / })
    await removes.last().click()
  }
  const addOne = async () => {
    await page.getByRole('button', { name: 'Add column' }).click()
  }

  // reconcile the count, starting from however many placeholders there are
  let current = await page.getByRole('textbox', { name: 'Name' }).count()
  while (current > columns.length) {
    await removeLast()
    current -= 1
  }
  while (current < columns.length) {
    await addOne()
    current += 1
  }

  for (const [index, column] of columns.entries()) {
    await page.locator(`#spec-name-${index}`).fill(column.name)
    await page.locator(`#spec-type-${index}`).selectOption(column.type)
    if (column.type === 'categorical') {
      await page.locator(`#spec-cats-${index}`).fill(column.value)
    } else {
      const [min, max] = column.value.split(',')
      await page.locator(`#spec-min-${index}`).fill(min)
      await page.locator(`#spec-max-${index}`).fill(max)
    }
  }
}

/** The dataset history entry for a version, by its number. */
export function versionNode(page: Page, number: number): Locator {
  return page.getByText(`Version ${number}`, { exact: true })
}

/** Open a step for a project. */
export async function gotoStep(page: Page, project: string, step: string): Promise<void> {
  await page.goto(`/projects/${await projectId(page, project)}/${step}`)
}

/** Resolve a Project's id through the API the page itself uses. */
export async function projectId(page: Page, name: string): Promise<string> {
  const response = await page.request.get('/api/projects')
  const projects = (await response.json()) as Array<{ id: string; name: string }>
  const found = projects.find((p) => p.name === name)
  if (!found) {
    throw new Error(
      `no Project named ${name}; the app has: ${projects.map((p) => p.name).join(', ') || '(none)'}`,
    )
  }
  return found.id
}

/** Open a step and wait for its heading, so the next step starts from a real page. */
export async function openStep(page: Page, project: string, step: string, heading: string) {
  await page.goto(`/projects/${await projectId(page, project)}/${step}`)
  await expect(page.getByText(heading, { exact: true }).first()).toBeVisible()
}

/**
 * Wait until a job-driven panel has finished.
 *
 * `done` is the text that step actually prints when it is over, rather than a
 * guess at a generic "finished" marker: Generate says how many Provider calls it
 * made, Label says how many rows it labeled, and a Training Run is finished when
 * the leaderboard is on screen. A regex here that does not match the real copy
 * makes the suite fail on a *successful* step, which is the worst possible
 * failure — it looks like a product bug and is not.
 */
export async function waitForJob(
  page: Page,
  panel: RegExp,
  done: RegExp,
  timeout = 120_000,
): Promise<void> {
  await expect(page.getByText(panel).first()).toBeVisible({ timeout: 30_000 })
  await expect(page.getByText(done).first()).toBeVisible({ timeout })
}

/** Acknowledge every outstanding warning on the panel in view. */
export async function acknowledgeAll(page: Page): Promise<number> {
  let count = 0
  // The panel re-renders after each acknowledgement, so re-query every time.
  // This used to `break` unconditionally on the first pass, which meant it could
  // only ever acknowledge one warning however many were outstanding — and the
  // `12` bound was dead. A step that raises two then failed its caller at
  // `toBeEnabled()` for a reason that had nothing to do with what it was testing.
  for (;;) {
    const ack = page.getByRole('button', { name: 'Acknowledge' }).first()
    if (!(await ack.isVisible().catch(() => false))) return count
    await ack.click()
    const save = page.getByRole('button', { name: 'Record Acknowledgement' })
    await save.waitFor({ state: 'visible', timeout: 10_000 })
    await save.click()
    count += 1
    await expect(page.getByText('Everything here has been read.')).toBeVisible({ timeout: 10_000 })
  }
}

/** Describe the page if something is missing, so a failure is diagnosable. */
export async function pageSummary(page: Page): Promise<string> {
  const text = await page.locator('body').innerText()
  return text.replace(/\n{2,}/g, '\n').slice(0, 1200)
}
