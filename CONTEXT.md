# Dat Distiller

A local, single-user workbench for responsibly producing classification and regression models: generate synthetic data, label it with Jev, and train models, with the provenance and quality of every row kept visible.

## Language

### Data

**Project**:
A named workspace that holds a chain of Dataset Versions and the Training Runs built from them.
_Avoid_: Workspace, session, job

**Dataset Version**:
An immutable table within a Project. Every step (upload, Generation, Labeling, review) produces a new Dataset Version from exactly one parent rather than changing an existing one. Relabeling an older version therefore starts a new branch.
_Avoid_: CSV, file, dataset (when a specific version is meant)

**Provenance**:
The per-row record of where a row and each of its labels came from: uploaded, synthetic (and which Generation Mode and Provider made it), Jev-labeled (with confidence), or human-reviewed.
_Avoid_: Lineage, source, origin

**Profile**:
A statistical description of a table's columns: types, distributions, category frequencies, missingness, and correlations. It is built from an uploaded sample or from Column Specs.
_Avoid_: Schema, stats, summary

**Column Spec**:
A user-declared column (name, type, optional constraints and description), used when there is no sample to learn a Profile from.
_Avoid_: Field definition, header

### Generation

**Generation**:
The step that produces synthetic rows resembling a description and a Profile.
_Avoid_: Synthesis, augmentation, sampling

**Generation Mode**:
How Generation produces rows: _statistical_ (a fitted statistical synthesizer), _llm_ (a Provider writes rows), or _hybrid_ (statistical for structured columns, Provider for free-text columns). The default is hybrid.
_Avoid_: Strategy, engine

**Balance Target**:
Proportions the user requests for a categorical column during Generation (for example 50/50 on `churned`).
_Avoid_: Stratification, oversampling

**Fidelity Report**:
The comparison of generated rows against the Profile (distribution tests, correlation drift, near-copy and duplicate counts).
_Avoid_: Validation report, quality score

**Provider**:
Anything that turns a prompt into structured text: a local CLI agent (Codex, Claude Code, OpenCode) or OpenRouter. Providers generate text only; they never label with Jev.
_Avoid_: Agent, LLM, backend, model (a Model is something the user trains)

### Labeling

**Jev**:
TypeSafe's hosted evaluation model. It reads a State and answers Jev Questions.
_Avoid_: Labeler, classifier, the model

**Labeling**:
The step that runs a Jev Question over every row of a Dataset Version and writes the answer into a new Label Column.
_Avoid_: Annotation, tagging, enrichment

**Jev Question**:
A typed question put to Jev: _Noul_ (yes/no, answered with a 0–1 confidence), _Choice_ (one of several named criteria, with a probability for each), or _Score_ (an ordinal level on a described scale). The user writes it, optionally from a draft suggested by a Provider.
_Avoid_: Prompt, result type, field type

**State**:
The text Jev reads for one row: the user-selected columns of that row, written out as `column: value` lines.
_Avoid_: Input, prompt, row text

**Label Column**:
A column added by Labeling that holds Jev's answer for each row. It comes with sibling confidence columns (one per option for Choice), which are also eligible as a Target. One Labeling run can ask several Jev Questions and so add several Label Columns.
_Avoid_: Result field, output column, target (until it is chosen for training)

**Review Queue**:
The low-confidence labels waiting for a human to accept or override them before training.
_Avoid_: Inbox, flagged rows

### Training

**Training Run**:
One attempt to fit one or more Model types to a chosen Target on a Dataset Version, together with its metrics and checks.
_Avoid_: Experiment, job, fit

**Target**:
The column a Training Run learns to predict. Its Task Type is _classification_ or _regression_.
_Avoid_: Label (ambiguous with Label Column), y, output

**Model**:
A trained estimator produced by a Training Run (for example an SVM, a Random Forest, or a neural net).
_Avoid_: Classifier (because regression is also supported), Provider

**Sensitive Attribute**:
A user-chosen column whose groups are compared in the Fairness Report.
_Avoid_: Protected class, demographic column

**Fairness Report**:
Per-group metrics for a Model, split by a Sensitive Attribute.
_Avoid_: Bias report

**Model Bundle**:
The downloadable package for a Model: the fitted estimator, its preprocessing, its class mapping and metadata, and its Model Card.
_Avoid_: Export, artifact, checkpoint

### Safeguards

**Check**:
A responsible-use safeguard evaluated at the end of a step (for example PII found, near-copies, class imbalance, leakage, unreviewed labels, fairness gaps), with severity _info_ or _warning_. A Check never blocks; a warning must be Acknowledged before continuing.
_Avoid_: Validation, guardrail, lint

**Acknowledgement**:
The user's explicit decision to continue despite a warning Check. It is recorded in the relevant Card.
_Avoid_: Override, dismissal

**Card**:
An exportable summary of how a Dataset Version (Dataset Card) or a Model (Model Card) was produced, including its Provenance, checks, and known limitations.
_Avoid_: Datasheet, report
