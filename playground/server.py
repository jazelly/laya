"""FastAPI UI that calls the same local Laya SDK runtime as the CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from playground.runtime import DecisionRuntime, PlaygroundInputError


class PredictRequest(BaseModel):
    """One request emitted by the browser form."""

    state: Union[str, Dict[str, Any], List[Any]]
    questions: Dict[str, Dict[str, Any]]
    checkpoint: str = Field(default="english")
    device: str = Field(default="mps")
    save: bool = Field(default=True)


RUNTIME = DecisionRuntime()
ASSET_DIRECTORY = Path(__file__).resolve().parents[1] / "assets"
app = FastAPI(title="Laya Local Playground", version="0.1.0")
app.mount("/assets", StaticFiles(directory=ASSET_DIRECTORY), name="assets")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Serve the one-page local Playground without a separate frontend build."""
    return HTML_PAGE


@app.get("/api/health")
def health() -> Dict[str, str]:
    """Confirm that the API and local SQLite schema are ready."""
    RUNTIME.initialize()
    return {"status": "ready", "database": str(RUNTIME.database_path)}


@app.post("/api/predict")
def predict(request: PredictRequest) -> Dict[str, Any]:
    """Call the shared SDK runtime from the browser UI."""
    try:
        return RUNTIME.predict(
            state=request.state,
            questions=request.questions,
            checkpoint=request.checkpoint,
            device=request.device,
            save=request.save,
        )
    except (PlaygroundInputError, RuntimeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/history")
def history(limit: int = 10) -> List[Dict[str, Any]]:
    """Return recent, intentionally saved local runs for inspection."""
    try:
        return RUNTIME.history(limit=limit)
    except PlaygroundInputError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def main() -> None:
    """Run the local UI on loopback only."""
    import uvicorn

    uvicorn.run("playground.server:app", host="127.0.0.1", port=8000, reload=False)


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Laya Local Playground</title>
  <link rel="icon" href="/assets/logo-mark.png">
  <style>
    :root {
      --paper: #f5f5f1;
      --surface: #ffffff;
      --ink: #1a1c1a;
      --muted: #656964;
      --line: #d9dcd6;
      --accent: #356847;
      --accent-ink: #f7fbf6;
      --code: #152018;
      --shadow: 0 18px 45px rgb(24 29 23 / 0.09);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: ui-rounded, "SF Pro Rounded", "Avenir Next", sans-serif;
      line-height: 1.45;
    }
    main { max-width: 1320px; margin: 0 auto; padding: clamp(24px, 5vw, 72px) clamp(18px, 4vw, 48px); }
    .masthead { display: flex; justify-content: space-between; gap: 24px; align-items: start; margin-bottom: 34px; }
    .brand { width: 150px; height: auto; }
    h1 { margin: 12px 0 8px; font-size: clamp(2rem, 4.5vw, 4.6rem); letter-spacing: -0.06em; line-height: 0.96; text-wrap: pretty; }
    .lead { max-width: 660px; color: var(--muted); font-size: 1.05rem; text-wrap: pretty; }
    .system { color: var(--muted); font-family: ui-monospace, "SFMono-Regular", Menlo, monospace; font-size: .77rem; text-align: right; }
    .grid { display: grid; grid-template-columns: minmax(0, 1.08fr) minmax(360px, .92fr); gap: 22px; }
    .panel { background: var(--surface); border: 1px solid var(--line); border-radius: 20px; box-shadow: var(--shadow); overflow: hidden; }
    .panel > header { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; padding: 20px 22px 15px; border-bottom: 1px solid var(--line); }
    h2 { margin: 0; font-size: 1.05rem; letter-spacing: -.02em; }
    .hint { color: var(--muted); font-size: .8rem; }
    form, .output { padding: 22px; }
    label { display: block; margin: 0 0 8px; font-size: .86rem; font-weight: 700; }
    textarea, select, button { font: inherit; }
    textarea, select { width: 100%; color: var(--ink); background: #fbfbf8; border: 1px solid var(--line); border-radius: 11px; }
    textarea { min-height: 192px; padding: 13px; resize: vertical; font-family: ui-monospace, "SFMono-Regular", Menlo, monospace; font-size: .8rem; line-height: 1.55; }
    select { padding: 10px 12px; }
    .field { margin-bottom: 18px; }
    .controls { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .toggle { display: flex; align-items: center; gap: 9px; color: var(--muted); font-size: .84rem; }
    input[type="checkbox"] { accent-color: var(--accent); width: 16px; height: 16px; }
    .actions { display: flex; align-items: center; justify-content: space-between; gap: 14px; margin-top: 20px; }
    button { cursor: pointer; border: 0; border-radius: 10px; padding: 11px 15px; font-weight: 700; }
    button.primary { background: var(--accent); color: var(--accent-ink); }
    button.secondary { background: transparent; color: var(--accent); border: 1px solid var(--line); }
    button:disabled { cursor: wait; opacity: .55; }
    #status { color: var(--muted); font-size: .82rem; }
    pre { margin: 0; padding: 16px; border-radius: 12px; background: var(--code); color: #e6efe5; overflow: auto; min-height: 390px; font: .78rem/1.55 ui-monospace, "SFMono-Regular", Menlo, monospace; white-space: pre-wrap; }
    .history { margin-top: 22px; }
    .history pre { min-height: 80px; max-height: 250px; }
    @media (max-width: 820px) {
      .masthead, .panel > header { align-items: start; flex-direction: column; }
      .system { text-align: left; }
      .grid { grid-template-columns: 1fr; }
    }
    @media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto !important; transition: none !important; } }
  </style>
</head>
<body>
  <main>
    <header class="masthead">
      <div>
        <img class="brand" src="/assets/logo-lockup.svg" alt="Laya">
        <h1>Local decision surface.</h1>
        <p class="lead">A loopback-only Playground. The form calls the same Python SDK runtime as the CLI, with PyTorch MPS selected for this Apple Silicon machine.</p>
      </div>
      <p class="system">Browser UI → /api/predict → Laya SDK → MPS<br>Optional local history → SQLite</p>
    </header>

    <section class="grid" aria-label="Laya Playground">
      <section class="panel">
        <header><h2>Decision request</h2><span class="hint">JSON in, typed answers out</span></header>
        <form id="prediction-form">
          <div class="field">
            <label for="state">State</label>
            <textarea id="state" spellcheck="false">{
  "message": "I was charged twice for invoice 4411. Please refund the duplicate today.",
  "account_tier": "enterprise"
}</textarea>
          </div>
          <div class="field">
            <label for="questions">Typed questions</label>
            <textarea id="questions" spellcheck="false">{
  "department": {
    "type": "choice",
    "instructions": "Which department should handle this request?",
    "criteria": {
      "billing": "Invoices, payments, and refunds",
      "technical": "Software bugs and outages",
      "sales": "Pricing and new purchases"
    }
  },
  "refund_requested": {
    "type": "noul",
    "instructions": "Does the customer explicitly request a refund?"
  },
  "urgency": {
    "type": "score",
    "instructions": "How urgent is this request?",
    "criteria": ["not urgent", "soon", "urgent"]
  }
}</textarea>
          </div>
          <div class="controls">
            <div class="field">
              <label for="checkpoint">Checkpoint</label>
              <select id="checkpoint"><option value="english">English</option><option value="multilingual">Multilingual</option><option value="typed-decisions">Typed decisions</option></select>
            </div>
            <div class="field">
              <label for="device">Device</label>
              <select id="device"><option value="mps">Apple GPU (MPS)</option><option value="cpu">CPU</option><option value="auto">Auto</option></select>
            </div>
          </div>
          <label class="toggle" for="save"><input id="save" type="checkbox" checked> Save this request and result in local SQLite</label>
          <div class="actions"><span id="status" aria-live="polite">Ready. The first checkpoint load can take a few minutes.</span><button class="primary" id="run" type="submit">Run decision</button></div>
        </form>
      </section>

      <section class="panel">
        <header><h2>SDK result</h2><span class="hint">raw response</span></header>
        <div class="output">
          <pre id="result" aria-live="polite">Run a decision to see the returned SDK payload.</pre>
          <div class="history">
            <div class="actions"><h2>SQLite history</h2><button class="secondary" id="load-history" type="button">Refresh</button></div>
            <pre id="history">No history loaded.</pre>
          </div>
        </div>
      </section>
    </section>
  </main>
  <script>
    const form = document.getElementById("prediction-form");
    const runButton = document.getElementById("run");
    const status = document.getElementById("status");
    const result = document.getElementById("result");
    const history = document.getElementById("history");

    function parseJson(id, label) {
      try { return JSON.parse(document.getElementById(id).value); }
      catch (error) { throw new Error(label + " must be valid JSON: " + error.message); }
    }

    async function loadHistory() {
      history.textContent = "Loading local history…";
      const response = await fetch("/api/history?limit=10");
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Could not load SQLite history.");
      history.textContent = JSON.stringify(payload, null, 2);
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      runButton.disabled = true;
      status.textContent = "Running the shared Laya SDK runtime…";
      try {
        const payload = {
          state: parseJson("state", "State"),
          questions: parseJson("questions", "Typed questions"),
          checkpoint: document.getElementById("checkpoint").value,
          device: document.getElementById("device").value,
          save: document.getElementById("save").checked
        };
        const response = await fetch("/api/predict", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(payload) });
        const responsePayload = await response.json();
        if (!response.ok) throw new Error(responsePayload.detail || "Prediction failed.");
        result.textContent = JSON.stringify(responsePayload, null, 2);
        status.textContent = "Completed on " + responsePayload.device + " in " + responsePayload.elapsed_ms + " ms.";
        if (payload.save) await loadHistory();
      } catch (error) {
        result.textContent = error.message;
        status.textContent = "Request needs attention.";
      } finally {
        runButton.disabled = false;
      }
    });

    document.getElementById("load-history").addEventListener("click", async () => {
      try { await loadHistory(); }
      catch (error) { history.textContent = error.message; }
    });
  </script>
</body>
</html>"""


if __name__ == "__main__":
    main()
