/**
 * The Label step's data layer: Jev Questions, the State column picker,
 * preview, estimate, and the resumable run.
 *
 * Question specs are validated here with the *same* rules the backend applies
 * (`parse_question`), so the builder can refuse a half-typed question with the
 * reason rather than discovering it at run time. `serializeState` mirrors
 * `jev.serialize_state` exactly — the live preview the user sees has to be the
 * text Jev will actually receive, character for character.
 */

import { useMutation } from '@tanstack/react-query'
import { apiPost } from '@/api/client'

// -- Jev Questions ------------------------------------------------------------

export const QUESTION_TYPES = ['noul', 'choice', 'score'] as const
export type QuestionType = (typeof QUESTION_TYPES)[number]

export interface ChoiceCriterion {
  key: string
  description: string
}

export interface JevQuestionDraft {
  type: QuestionType
  name: string
  instructions: string
  /** Choice only, in the order shown. */
  criteria: ChoiceCriterion[]
  /** Score only, lowest to highest. */
  levels: string[]
}

export function emptyQuestion(index: number): JevQuestionDraft {
  return {
    type: 'noul',
    name: `question_${index + 1}`,
    instructions: '',
    criteria: [
      { key: 'yes_category', description: '' },
      { key: 'no_category', description: '' },
    ],
    levels: ['', ''],
  }
}

export function questionToSpec(question: JevQuestionDraft): Record<string, unknown> {
  const spec: Record<string, unknown> = {
    type: question.type,
    name: question.name.trim(),
    instructions: question.instructions.trim(),
  }
  if (question.type === 'choice') {
    // the builder edits a list; the API wants a key -> description map
    const criteria: Record<string, string> = {}
    for (const { key, description } of question.criteria) {
      const clean = key.trim()
      if (clean) criteria[clean] = description.trim()
    }
    spec.criteria = criteria
  }
  if (question.type === 'score') {
    spec.levels = question.levels.map((level) => level.trim()).filter(Boolean)
  }
  return spec
}

/** Everything wrong with a draft, in the order the user will fix it. */
/**
 * The sibling column one Choice option writes its probability to.
 *
 * Mirrors `label._option_column` in the backend, which is the definition that
 * actually decides where a probability lands. The two must agree: if they
 * drift, the UI would validate a set of options the backend then refuses, or
 * worse, accept one whose probability is written to a column nobody expected.
 */
export function optionColumn(option: string): string {
  return `p_${option.replace(/[^a-zA-Z0-9_]+/g, '_').replace(/^_+|_+$/g, '').toLowerCase()}`
}

export function questionErrors(question: JevQuestionDraft): string[] {
  const errors: string[] = []
  const name = question.name.trim()
  if (!name) errors.push('a name is required')
  else if (!/^[a-z][a-z0-9_]*$/.test(name)) {
    errors.push('the name must be snake_case: lowercase, digits and underscores only')
  }
  if (!question.instructions.trim()) {
    errors.push('tell Jev what to decide — instructions are required')
  }
  if (question.type === 'choice') {
    const keys = question.criteria.map((c) => c.key.trim()).filter(Boolean)
    if (keys.length < 2) errors.push('a Choice needs at least two options')
    const seen = new Set<string>()
    for (const key of keys) {
      if (seen.has(key)) errors.push(`option "${key}" is duplicated`)
      seen.add(key)
    }
    // Two options that differ only in case or punctuation would share one
    // probability column, and the second write would silently overwrite the
    // first. The backend refuses this too; catching it here means the person
    // editing the Jev Question hears about it before they run anything.
    const columns = new Map<string, string>()
    for (const key of keys) {
      const column = optionColumn(key)
      const first = columns.get(column)
      if (first !== undefined) {
        errors.push(`options "${first}" and "${key}" both become the column ${column}`)
      } else {
        columns.set(column, key)
      }
    }
  }
  if (question.type === 'score') {
    const levels = question.levels.map((l) => l.trim()).filter(Boolean)
    if (levels.length < 2) errors.push('a Score needs at least two levels')
  }
  return errors
}

/** A spec ready to send, or null while the draft is still incomplete. */
export function questionSpecOrNull(
  question: JevQuestionDraft,
): Record<string, unknown> | null {
  return questionErrors(question).length === 0 ? questionToSpec(question) : null
}

/** The Label Columns Labeling will add for these questions. */
export function labelColumns(specs: Array<Record<string, unknown>>): string[] {
  const columns: string[] = []
  for (const spec of specs) {
    const name = String(spec.name)
    columns.push(name, `${name}__confidence`)
    if (spec.type === 'choice') {
      for (const key of Object.keys((spec.criteria ?? {}) as Record<string, string>)) {
        columns.push(`${name}__p_${key.replace(/[^a-zA-Z0-9_]+/g, '_').replace(/^_+|_+$/g, '').toLowerCase()}`)
      }
    }
    if (spec.type === 'score') columns.push(`${name}__probabilities`)
  }
  return columns
}

// -- the State picker ---------------------------------------------------------

/**
 * Render one row as Jev will see it. Mirrors `jev.serialize_state` exactly:
 * columns in the chosen order, whitespace inside a value collapsed to single
 * spaces (so one row stays one logical block), missing values left empty, and
 * each line right-trimmed. If this drifts from the backend the live preview
 * becomes a lie — it is pinned by a test that compares against real backend
 * output.
 */
export function serializeState(
  row: Record<string, unknown>,
  columns: string[],
): string {
  return columns
    .map((column) => {
      const value = row[column]
      const raw = value === null || value === undefined ? '' : String(value)
      const text = raw.trim().replace(/\s+/g, ' ')
      return `${column}: ${text}`.trimEnd()
    })
    .join('\n')
}

// -- request bodies -----------------------------------------------------------

export interface LabelBody {
  version_id: string
  questions: Array<Record<string, unknown>>
  state_columns: string[]
  jev_model?: string
  preview_rows?: number
}

export interface LabelPreviewRow {
  state: string
  answers: Record<string, { answer?: string; confidence?: number; probabilities?: Record<string, number> }>
}

export interface LabelPreview {
  state_columns: string[]
  rows: LabelPreviewRow[]
}

export interface LabelEstimate {
  rows: number
  estimated_jev_calls: number
  uses_jev: boolean
}

export interface LabelRunResult {
  version_id: string
  labeled_rows: number
  failed_rows: number[]
  failed_count: number
  label_columns: string[]
}

export function useLabelPreview() {
  return useMutation({
    mutationFn: (body: LabelBody) =>
      apiPost<LabelPreview>('/label/preview', { ...body, preview_rows: 5 }),
  })
}

export function useLabelEstimate() {
  return useMutation({
    mutationFn: (body: LabelBody) => apiPost<LabelEstimate>('/label/estimate', body),
  })
}

export function useStartLabelRun() {
  return useMutation({
    mutationFn: (body: LabelBody) =>
      apiPost<{ id: string }>('/label/run', body).then((r) => r.id),
  })
}
