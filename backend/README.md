# Dat Distiller backend

uv-managed Python 3.11–3.12 package (`dat_distiller`): FastAPI app plus the
`dat-distiller` command.

```sh
uv sync                 # install the project (dev group included)
uv run pytest           # tests
uv run dat-distiller serve --port 8756
```

`serve` listens on `127.0.0.1:8756` by default (`--host`, `--port`) and serves
the built frontend from the repo root's `frontend/dist` on the same port,
when that build exists.

Optional extras stay out of the dev environment; install what you need:

```sh
uv sync --extra torch        # or --extra tensorflow, --extra presidio, --extra all
```

`GET /api/health` reports which of them are installed.
