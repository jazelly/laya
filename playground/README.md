# Local Playground

The UI and CLI call the same `DecisionRuntime`, which loads the Laya Python SDK directly. The
only persistence is an optional SQLite file at `playground/.local/laya_playground.sqlite3`.
It is ignored by Git because it can include the states and results entered locally.

## Apple Silicon setup

Run these commands from the repository root:

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -e ".[playground]"
.venv/bin/python -m playground.cli init-db
```

Confirm that PyTorch can see Metal before relying on MPS:

```bash
.venv/bin/python -c 'import torch; print(torch.backends.mps.is_available())'
```

## UI

```bash
.venv/bin/python -m playground.server
```

Open `http://127.0.0.1:8000`. The UI sends a JSON request to `/api/predict`; that endpoint calls
the shared `DecisionRuntime`, which invokes `laya.load(..., device="mps")` and then
`agent.predict(state, questions)`. The first request downloads the chosen public checkpoint.

## CLI

The sample files exercise the exact same runtime as the browser:

```bash
.venv/bin/python -m playground.cli predict \
  --state-file playground/samples/state.json \
  --questions-file playground/samples/questions.json \
  --checkpoint english \
  --device mps \
  --save

.venv/bin/python -m playground.cli history --limit 10
```

Use `--state 'plain text is also accepted'` instead of `--state-file` when a JSON object is not
needed. `--save` is opt-in on the CLI; the UI makes local saving explicit with its checkbox.
