"""Command-line entry point for the local Laya Playground."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from playground.runtime import CHECKPOINTS, DEFAULT_DATABASE_PATH, DecisionRuntime, PlaygroundInputError


def read_json_file(path: Path, label: str) -> Any:
    """Read a UTF-8 JSON file and give callers a clear path-specific error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise PlaygroundInputError("%s file does not exist: %s" % (label, path)) from error
    except json.JSONDecodeError as error:
        raise PlaygroundInputError("%s file is not valid JSON: %s" % (label, error)) from error


def parse_state_text(value: str) -> Any:
    """Use JSON state when possible and otherwise pass plain text to the SDK."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def build_parser() -> argparse.ArgumentParser:
    """Build the small, explicit command surface for the Playground."""
    parser = argparse.ArgumentParser(description="Local Laya SDK Playground")
    parser.add_argument(
        "--database",
        default=str(DEFAULT_DATABASE_PATH),
        help="SQLite database path (default: %(default)s)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("init-db", help="Create the local SQLite database and decision_runs table")

    predict = commands.add_parser("predict", help="Run one real Laya SDK prediction")
    state_source = predict.add_mutually_exclusive_group(required=True)
    state_source.add_argument("--state", help="Plain text or JSON state")
    state_source.add_argument("--state-file", type=Path, help="UTF-8 JSON state file")
    predict.add_argument("--questions-file", type=Path, required=True, help="UTF-8 JSON questions file")
    predict.add_argument("--checkpoint", choices=CHECKPOINTS.keys(), default="english")
    predict.add_argument("--device", choices=("mps", "cuda", "cpu", "auto"), default="mps")
    predict.add_argument("--save", action="store_true", help="Save this run to local SQLite")

    history = commands.add_parser("history", help="Read locally saved runs")
    history.add_argument("--limit", type=int, default=10)
    return parser


def main() -> None:
    """Execute the selected local Playground command."""
    args = build_parser().parse_args()
    runtime = DecisionRuntime(database_path=Path(args.database))

    if args.command == "init-db":
        runtime.initialize()
        print("SQLite ready: %s" % runtime.database_path)
        return

    if args.command == "history":
        print(json.dumps(runtime.history(args.limit), ensure_ascii=False, indent=2))
        return

    if args.state_file is not None:
        state = read_json_file(args.state_file, "state")
    else:
        state = parse_state_text(args.state)
    questions = read_json_file(args.questions_file, "questions")
    if not isinstance(questions, dict):
        raise PlaygroundInputError("questions file must contain a JSON object")
    response: Dict[str, Any] = runtime.predict(
        state=state,
        questions=questions,
        checkpoint=args.checkpoint,
        device=args.device,
        save=args.save,
    )
    print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
