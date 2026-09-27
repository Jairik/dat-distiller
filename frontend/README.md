# Dat Distiller — frontend

Vite + React + TypeScript app, served by `dat-distiller serve` from `dist/`.

## Commands (pnpm)

| Command           | What it does                                    |
| ----------------- | ----------------------------------------------- |
| `pnpm dev`        | Dev server; proxies `/api` → `127.0.0.1:8756`   |
| `pnpm build`      | `tsc -b` typecheck + production build into `dist/` |
| `pnpm typecheck`  | TypeScript only                                 |
| `pnpm test`       | Vitest (jsdom + Testing Library)                |

Run the backend alongside with `dat-distiller serve` from `backend/` (uv).

## Conventions

- **Theme**: dark first. All design tokens are CSS variables in
  [`src/index.css`](src/index.css); `:root` carries the dark palette and a
  `.light` class on `<html>` opts into the light variant. shadcn/ui (`new-york`,
  CSS variables) primitives live in `src/components/ui/`.
- **Motion**: use the primitives in `src/components/motion/` (`PageIn`,
  `StaggerGroup`/`StaggerItem`, `HoverPress`, `AnimatedNumber`). They all
  collapse to instant/opacity-only under `prefers-reduced-motion`.
- **API**: talk to the backend through `src/api/client.ts`
  (`apiGet`/`apiPost`/`apiDelete`, errors as `ApiError` with FastAPI `detail`),
  via TanStack Query.
- Tests use Vitest + Testing Library; `src/test/setup.ts` stubs
  `matchMedia`, and `src/test/reduced-motion.ts` flips the reduced-motion
  flag for tests.
