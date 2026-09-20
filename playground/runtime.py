"""Shared Laya runtime and SQLite history used by the Playground UI and CLI."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

import laya


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "playground" / ".local" / "laya_playground.sqlite3"
CHECKPOINTS: Dict[str, Tuple[str, Optional[str]]] = {
    "english": ("convaiinnovations/laya", None),
    "multilingual": ("convaiinnovations/laya", "multilingual"),
    "typed-decisions": ("convaiinnovations/laya", "typed-decisions"),
}
State = Union[str, Dict[str, Any], List[Any]]
Questions = Dict[str, Dict[str, Any]]


class PlaygroundInputError(ValueError):
    """Raised when a Playground request cannot be represented by the Laya SDK."""


def resolve_device(requested: str) -> str:
    """Resolve an explicit Playground device request without silently downgrading it."""
    normalized = requested.lower().strip()
    if normalized == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if normalized == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS is unavailable in this PyTorch installation.")
        return "mps"
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this PyTorch installation.")
        return "cuda"
    if normalized == "cpu":
        return "cpu"
    raise PlaygroundInputError("device must be one of: mps, cuda, cpu, auto")


def validate_state(state: State) -> None:
    """Validate the state forms accepted by ``laya.Agent.predict``."""
    if not isinstance(state, (str, dict, list)):
        raise PlaygroundInputError("state must be a JSON object, JSON array, or text string")


def validate_questions(questions: Questions) -> None:
    """Validate the compact typed-question contract before loading a model."""
    if not isinstance(questions, dict) or not questions:
        raise PlaygroundInputError("questions must be a non-empty JSON object")

    for question_id, definition in questions.items():
        if not isinstance(question_id, str) or not question_id:
            raise PlaygroundInputError("each question id must be a non-empty string")
        if not isinstance(definition, dict):
            raise PlaygroundInputError("question %r must be a JSON object" % question_id)
        question_type = definition.get("type")
        if question_type not in ("choice", "score", "noul"):
            raise PlaygroundInputError("question %r has an unsupported type" % question_id)
        if not isinstance(definition.get("instructions"), str):
            raise PlaygroundInputError("question %r needs string instructions" % question_id)

        criteria = definition.get("criteria")
        if question_type == "choice":
            if not isinstance(criteria, (dict, list)) or not criteria:
                raise PlaygroundInputError("choice question %r needs non-empty criteria" % question_id)
        if question_type == "score":
            if not isinstance(criteria, list) or len(criteria) < 2:
                raise PlaygroundInputError("score question %r needs at least two ordered criteria" % question_id)


@dataclass
class SQLiteHistory:
    """Persist optional local Playground runs in one small SQLite table."""

    database_path: Path

    def initialize(self) -> None:
        """Create the local database and table if they do not yet exist."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS decision_runs (
                    id INTEGER PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    checkpoint TEXT NOT NULL,
                    device TEXT NOT NULL,
                    elapsed_ms REAL NOT NULL,
                    state_json TEXT NOT NULL,
                    questions_json TEXT NOT NULL,
                    result_json TEXT NOT NULL
                )
                """
            )

    def record(
        self,
        checkpoint: str,
        device: str,
        elapsed_ms: float,
        state: State,
        questions: Questions,
        result: Dict[str, Any],
    ) -> int:
        """Save one completed SDK prediction and return its SQLite row id."""
        self.initialize()
        created_at = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.database_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO decision_runs (
                    created_at, checkpoint, device, elapsed_ms,
                    state_json, questions_json, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    checkpoint,
                    device,
                    elapsed_ms,
                    json.dumps(state, ensure_ascii=False),
                    json.dumps(questions, ensure_ascii=False),
                    json.dumps(result, ensure_ascii=False),
                ),
            )
        return int(cursor.lastrowid)

    def recent(self, limit: int) -> List[Dict[str, Any]]:
        """Return the newest local runs, including the original state and SDK result."""
        if limit < 1:
            raise PlaygroundInputError("history limit must be at least 1")
        self.initialize()
        with sqlite3.connect(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT id, created_at, checkpoint, device, elapsed_ms,
                       state_json, questions_json, result_json
                FROM decision_runs
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "id": row[0],
                "created_at": row[1],
                "checkpoint": row[2],
                "device": row[3],
                "elapsed_ms": row[4],
                "state": json.loads(row[5]),
                "questions": json.loads(row[6]),
                "result": json.loads(row[7]),
            }
            for row in rows
        ]


@dataclass
class DecisionRuntime:
    """Keep one requested checkpoint resident for UI and CLI predictions."""

    database_path: Path = DEFAULT_DATABASE_PATH
    _history: SQLiteHistory = field(init=False)
    _agent: Any = field(default=None, init=False, repr=False)
    _cache_key: Optional[Tuple[str, str]] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Normalize a caller-provided database path once."""
        self.database_path = Path(self.database_path).expanduser()
        self._history = SQLiteHistory(self.database_path)

    def initialize(self) -> None:
        """Initialize the local SQLite database without loading model weights."""
        self._history.initialize()

    def _get_agent(self, checkpoint: str, requested_device: str) -> Tuple[Any, str]:
        """Load a checkpoint only when the selected checkpoint or device changes."""
        if checkpoint not in CHECKPOINTS:
            raise PlaygroundInputError(
                "checkpoint must be one of: %s" % ", ".join(CHECKPOINTS.keys())
            )

        device = resolve_device(requested_device)
        cache_key = (checkpoint, device)
        if self._agent is None or self._cache_key != cache_key:
            model_id, subfolder = CHECKPOINTS[checkpoint]
            agent = laya.load(model_id, device=device, subfolder=subfolder)
            actual_device = str(agent.device)
            if actual_device != device:
                raise RuntimeError(
                    "Laya loaded on %s although %s was requested." % (actual_device, device)
                )
            self._agent = agent
            self._cache_key = cache_key
        return self._agent, device

    def predict(
        self,
        state: State,
        questions: Questions,
        checkpoint: str = "english",
        device: str = "mps",
        save: bool = True,
    ) -> Dict[str, Any]:
        """Run one real Laya SDK prediction and optionally persist it locally."""
        validate_state(state)
        validate_questions(questions)
        self.initialize()
        agent, actual_device = self._get_agent(checkpoint, device)

        started = time.perf_counter()
        result = agent.predict(state, questions)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        response: Dict[str, Any] = {
            "checkpoint": checkpoint,
            "device": actual_device,
            "elapsed_ms": elapsed_ms,
            "result": result,
        }
        if save:
            response["run_id"] = self._history.record(
                checkpoint=checkpoint,
                device=actual_device,
                elapsed_ms=elapsed_ms,
                state=state,
                questions=questions,
                result=result,
            )
        return response

    def history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Read previously persisted local Playground runs."""
        return self._history.recent(limit)
