# jev-distiller
Simple web interface for distilling jev for creating smaller classification models

## Running it

```bash
cd backend && uv run dat-distiller serve
```

The app serves its own frontend, so there is no second web server. Settings live in
`~/.config/dat-distiller` (override with `DAT_DISTILLER_DATA_DIR` / `DAT_DISTILLER_CONFIG_DIR`).

## Testing

| What | Command | Needs |
|---|---|---|
| Backend | `cd backend && uv run pytest -q` | nothing |
| Frontend | `cd frontend && npm test` | nothing |
| Types & build | `cd frontend && npm run build` | nothing |
| End to end | `cd frontend && npm run e2e` | a browser, once |

### End to end

`npm run e2e` runs a Playwright suite in a real Chromium against a real
`dat-distiller serve`, serving the real built `frontend/dist`. It walks the whole
flow — Project, generate from Column Specs, acknowledge Checks, Label, Review,
Train, Fairness Report, download the Model Bundle and both Cards — plus the
upload path.

**No network and no API keys.** The server is started with
`DAT_DISTILLER_FAKE_PROVIDERS=1` and `DAT_DISTILLER_FAKE_JEV=1`, so the Provider
and Jev are deterministic local fakes. `webServer` also runs
`uv sync --extra sklearn --extra lightgbm` and `vite build` first, because the
Train step has no runnable Models without the extras and the backend serves a
blank page without a build.

Two one-time prerequisites, the same kind as `npm install`:

```bash
cd frontend && npm install && npm run e2e:install   # downloads Chromium
```

The suite writes to a throwaway data directory (`.e2e-data/`, git-ignored) and
never touches your real Projects. `npm run e2e:ui` opens the Playwright UI for
watching a run.
