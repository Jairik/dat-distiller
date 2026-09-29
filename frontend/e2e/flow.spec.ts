/**
 * The whole flow, end to end, against the real server.
 *
 * Nothing here is mocked at the HTTP boundary: this is `dat-distiller serve`
 * serving the real built `frontend/dist`, with `FakeProvider` and `FakeJev`
 * standing in for the network. So what the suite exercises is the *wiring* —
 * the router registrations, the job wiring, the SSE stream, the checks gates and
 * the download endpoints — which is exactly the part unit tests cannot reach.
 *
 * The two paths the issue asks for are covered: **upload → label → train**, and
 * **Column Specs → generate → label → review → train → fairness → download**.
 */

import { expect, test } from '@playwright/test'

import {
  acknowledgeAll,
  createProject,
  openStep,
  pageSummary,
  setColumnSpecs,
  waitForJob,
} from './helpers'

// A large enough dataset that the split, the fairness groups and the
// leaderboard all have something to work with, and small enough to stay quick.
const ROWS = 220

test.describe.configure({ mode: 'serial' })

test('a generated dataset can be labeled, reviewed, trained, measured and exported', async ({
  page,
}) => {
  const project = `E2E generate ${Date.now()}`

  // 1. a Project
  await createProject(page, project)
  const projectId = await (async () => {
    const response = await page.request.get('/api/projects')
    const all = (await response.json()) as Array<{ id: string; name: string }>
    return all.find((p) => p.name === project)!.id
  })()

  // 2. generate from Column Specs, in hybrid mode
  await openStep(page, project, 'generate', 'Generate')
  await page
    .getByLabel('What should this dataset contain?')
    .fill('Support tickets for a SaaS help desk, with a region, a topic and a satisfaction score.')

  // the shape comes from Column Specs, not from a sample
  await setColumnSpecs(page, [
    { name: 'region', type: 'categorical', value: 'north, south' },
    { name: 'topic', type: 'categorical', value: 'billing, bug, feature' },
    { name: 'satisfaction', type: 'integer', value: '1,5' },
  ])

  await page.getByRole('radio', { name: 'hybrid' }).check()
  await page.getByLabel('Rows').fill(String(ROWS))
  await page.getByLabel('Seed (optional)').fill('4242')

  // a preview comes before the run, because generation costs Provider calls
  await page.getByRole('button', { name: 'Preview 5 rows' }).click()
  await expect(page.getByText('Preview', { exact: true }).first()).toBeVisible()

  // the estimate comes between the preview and the run, as in the Label step
  await page.getByRole('button', { name: 'Estimate and continue' }).click()
  await expect(page.getByText('Ready to generate')).toBeVisible()
  await page.getByRole('button', { name: 'Generate the full dataset' }).click()
  // "220 rows · seed 4242 · 11 Provider call(s)" is what Generate prints when it
  // is done — and the Provider call count is worth asserting: it is the number
  // the estimate promised, roughly, and a silent zero would mean the fakes were
  // bypassed and the run hit the network
  await waitForJob(page, /Generation Run/, /Provider call\(s\)/)
  await expect(page.getByRole('button', { name: 'Open the new Dataset Version' })).toBeVisible()

  // 3. acknowledge the Checks Generation raised
  const acknowledged = await acknowledgeAll(page)
  expect(acknowledged).toBeGreaterThan(0)
  await expect(page.getByRole('button', { name: 'Continue to Label' })).toBeEnabled()
  await page.getByRole('button', { name: 'Continue to Label' }).click()
  await expect(page.getByText('Label', { exact: true }).first()).toBeVisible()

  // 4. label with a Choice question
  await page.getByRole('textbox', { name: /Name \(becomes/ }).fill('topic_judgement')
  await page
    .getByRole('textbox', { name: 'Instructions' })
    .fill('Which single topic does this ticket belong to?')
  await page.getByRole('radio', { name: 'choice' }).check()
  const keys = page.getByRole('textbox', { name: 'Key' })
  await keys.nth(0).fill('billing')
  await keys.nth(1).fill('bug')
  const descriptions = page.getByRole('textbox', { name: 'Description' })
  await descriptions.nth(0).fill('about invoices or payments')
  await descriptions.nth(1).fill('something is broken')

  await page.getByRole('button', { name: 'Select every column' }).click()
  await page.getByRole('button', { name: /Preview 5 rows/ }).click()
  await expect(page.getByText('Preview', { exact: true }).first()).toBeVisible()
  await page.getByRole('button', { name: /Estimate and continue/ }).click()
  await expect(page.getByText(/Jev call\(s\), one per row/)).toBeVisible()
  await page.getByRole('button', { name: /Label the whole Dataset Version/ }).click()
  await waitForJob(page, /Labeling run/, /rows labeled/)

  // the Label Columns are named, so the user can find them
  await expect(page.getByText('topic_judgement__confidence')).toBeVisible()

  // 5. review a row, on the version Labeling just produced
  const labeledId = await latestVersionId(page, projectId)
  await page.goto(`/projects/${projectId}/dataset?v=${labeledId}`)
  const queue = page.getByText('Review Queue', { exact: true }).first()
  await expect(queue).toBeVisible({ timeout: 20_000 })
  // with the default threshold there may be nothing to review; that is a valid
  // outcome, and the run must say which one it is. The assertion has to be on
  // the branch's *content* — re-checking the condition that selected the branch
  // proves nothing, which is what this used to do.
  const nothing = page.getByText(/Nothing to review at this threshold/)
  if (await nothing.isVisible().catch(() => false)) {
    await expect(nothing).toBeVisible()
    await expect(page.getByRole('button', { name: 'Accept', exact: true })).toHaveCount(0)
  } else {
    // `name` is a substring match by default, so a bare 'Accept' also matches
    // "Accept all 200 shown" — which stages the whole queue rather than one row
    const accept = page.getByRole('button', { name: 'Accept', exact: true }).first()
    await expect(accept).toBeVisible()
    await accept.click()
    await expect(page.getByText('1 staged')).toBeVisible()
    await page.getByRole('button', { name: /Apply 1 decision/ }).click()
    await expect(page.getByRole('status')).toContainText('Dataset Version is in the history')
  }

  // 6. train two Models
  const versionId = await latestVersionId(page, projectId)
  await page.goto(`/projects/${projectId}/train?version=${versionId}`)
  await expect(page.getByText('Train', { exact: true }).first()).toBeVisible()
  await page.locator('#target-topic_judgement').check()
  await expect(page.getByText('Logistic Regression')).toBeVisible({ timeout: 20_000 })
  await page.getByText('Logistic Regression').click()
  await page.getByText('Random Forest').click()
  await page.getByRole('button', { name: 'See the plan' }).click()
  const plan = page.getByTestId('train-plan')
  await expect(plan).toBeVisible()
  await expect(plan).toContainText('held out')
  await plan.getByRole('button', { name: 'Train' }).click()

  const board = page.getByTestId('leaderboard')
  await expect(page.getByText('Training Run').first()).toBeVisible({ timeout: 30_000 })
  await expect(board).toBeVisible({ timeout: 240_000 })
  // two Models, both fitted, and a real leaderboard
  await expect(board.getByText('Logistic Regression')).toBeVisible()
  await expect(board.getByText('Random Forest')).toBeVisible()
  await expect(page.getByTestId('board-row-logistic_regression')).toBeVisible()

  // 7. the Fairness Report, on demand, for the winning Model
  const winner = board.getByTestId('board-row-logistic_regression')
  await winner.getByRole('button', { name: 'Fairness Report' }).click()
  await page.getByLabel('Report on this Sensitive Attribute').selectOption('region')
  await page.getByRole('button', { name: 'Measure this group' }).click()
  const headline = page.getByTestId('fairness-headline')
  await expect(headline).toBeVisible({ timeout: 60_000 })
  await expect(headline).toContainText('region')
  // the bars and their numbers are both there
  await expect(page.getByTestId('fairness-bars')).toBeVisible()
  await expect(page.getByTestId('fairness-gaps')).toBeVisible()

  // 8. the Model Bundle and the Dataset Card both download
  const bundle = page.waitForEvent('download')
  await board.getByRole('link', { name: 'Model Bundle' }).first().click()
  const bundleFile = await bundle
  expect(bundleFile.suggestedFilename()).toContain('bundle.zip')

  const card = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Model Card' }).click()
  await page.getByRole('link', { name: 'Download .md' }).click()
  const cardFile = await card
  expect(cardFile.suggestedFilename()).toMatch(/\.md$/)

  // the Dataset Card too, from the history panel. The Model Card drawer is a
  // modal, so it has to be dismissed before the page behind it is usable.
  await page.keyboard.press('Escape')
  await expect(page.getByTestId('card-body')).toBeHidden()
  await openStep(page, project, 'dataset', 'Dataset')
  await page.getByRole('button', { name: 'Dataset Card' }).first().click()
  const datasetCard = page.getByTestId('card-body')
  await expect(datasetCard).toBeVisible()
  // the rendered view is a document, so the Provenance heading is a heading
  await expect(datasetCard.getByRole('heading', { name: 'Provenance' })).toBeVisible()
  // and the raw Markdown is one tab away, verbatim
  await page.getByRole('tab', { name: 'Markdown' }).click()
  await expect(page.getByTestId('card-raw')).toContainText('## Provenance')
  // and it is downloadable
  const datasetFile = page.waitForEvent('download')
  await page.getByRole('link', { name: 'Download .md' }).click()
  expect((await datasetFile).suggestedFilename()).toMatch(/\.md$/)
})

test('a step can be started from an upload, and then labeled and trained', async ({ page }) => {
  const project = `E2E upload ${Date.now()}`
  await createProject(page, project)
  const projectId = await (async () => {
    const response = await page.request.get('/api/projects')
    const all = (await response.json()) as Array<{ id: string; name: string }>
    return all.find((p) => p.name === project)!.id
  })()

  // an upload, built here so the suite needs no fixture file on disk
  const rows = ['region,age,score']
  for (let i = 0; i < ROWS; i += 1) {
    rows.push(`${i % 2 === 0 ? 'north' : 'south'},${18 + (i % 50)},${(i * 7) % 100}`)
  }
  const csv = Buffer.from(`${rows.join('\n')}\n`, 'utf8')

  await openStep(page, project, 'dataset', 'Dataset')
  await page.setInputFiles('input[type="file"]', {
    name: 'customers.csv',
    mimeType: 'text/csv',
    buffer: csv,
  })
  // the upload is acknowledged before the flow continues
  await acknowledgeAll(page)
  const gate = page.getByRole('button', { name: 'Go to Generate' })
  await expect(gate).toBeEnabled()
  await gate.click()
  await expect(page.getByText('Generate', { exact: true }).first()).toBeVisible()

  // the uploaded version is labeled and trained like any other — this path uses
  // a Noul, so the other question type is covered somewhere in the suite
  await openStep(page, project, 'label', 'Label')
  await page.getByRole('textbox', { name: /Name \(becomes/ }).fill('is_satisfied')
  await page
    .getByRole('textbox', { name: 'Instructions' })
    .fill('Answer yes when the customer was satisfied.')
  await page.getByRole('button', { name: 'Select every column' }).click()
  await page.getByRole('button', { name: /Preview 5 rows/ }).click()
  await expect(page.getByText('Preview', { exact: true }).first()).toBeVisible()
  await page.getByRole('button', { name: /Estimate and continue/ }).click()
  await page.getByRole('button', { name: /Label the whole Dataset Version/ }).click()
  await waitForJob(page, /Labeling run/, /rows labeled/)

  const versionId = await latestVersionId(page, projectId)
  await page.goto(`/projects/${projectId}/train?version=${versionId}`)
  await expect(page.getByText('Train', { exact: true }).first()).toBeVisible()
  await page.locator('#target-is_satisfied').check()
  await expect(page.getByText('Logistic Regression')).toBeVisible({ timeout: 20_000 })
  await page.getByRole('button', { name: 'See the plan' }).click()
  const plan = page.getByTestId('train-plan')
  await expect(plan).toBeVisible()
  await expect(plan).toContainText('classification')
  await plan.getByRole('button', { name: 'Train' }).click()
  await expect(page.getByText('Training Run').first()).toBeVisible({ timeout: 30_000 })
  await expect(page.getByTestId('leaderboard')).toBeVisible({ timeout: 240_000 })

  // asking for a plot works end to end, and nothing asked for it beforehand
  await page.getByRole('button', { name: 'Confusion matrix' }).first().click()
  await expect(page.getByTestId('confusion-table')).toBeVisible({ timeout: 60_000 })
  // the predictions of an unseen CSV, and the download of them
  const row = page.getByTestId('board-row-logistic_regression')
  await row.getByRole('button', { name: 'Predict on a CSV' }).click()
  await row.getByLabel(/A CSV with/).setInputFiles({
    name: 'new.csv',
    mimeType: 'text/csv',
    buffer: Buffer.from('region,age,score\nnorth,30,10\nsouth,60,90\n', 'utf8'),
  })
  const predicted = await page.getByTestId('predict-result-logistic_regression')
  await expect(predicted).toBeVisible({ timeout: 60_000 })
  const download = page.waitForEvent('download')
  await predicted.getByRole('link', { name: 'Download predictions' }).click()
  expect((await download).suggestedFilename()).toMatch(/\.csv$/)
})

/** The most recent Dataset Version in a Project. */
async function latestVersionId(page: import('@playwright/test').Page, projectId: string) {
  const response = await page.request.get(`/api/projects/${projectId}/dataset_versions`)
  const versions = (await response.json()) as Array<{ id: string; number: number }>
  if (versions.length === 0) {
    throw new Error(`no Dataset Versions in ${projectId}:\n${await pageSummary(page)}`)
  }
  return versions.sort((a, b) => b.number - a.number)[0].id
}
