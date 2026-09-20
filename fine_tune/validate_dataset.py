"""Validate the Laya fine-tuning JSONL contract and export notebook-compatible rows."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


class DatasetValidationError(ValueError):
    """Raised for one malformed fine-tuning case."""


def expected_probability_keys(question: Dict[str, Any]) -> List[str]:
    """Return the exact probability keys required by one typed question."""
    question_type = question.get("type")
    criteria = question.get("criteria")
    if question_type == "choice":
        if isinstance(criteria, dict) and criteria:
            return list(criteria.keys())
        if isinstance(criteria, list) and criteria and all(isinstance(item, str) for item in criteria):
            return criteria
        raise DatasetValidationError("choice criteria must be a non-empty object or string list")
    if question_type == "noul":
        return ["false", "true"]
    if question_type == "score":
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise DatasetValidationError("score criteria must contain at least two ordered labels")
        return [str(index) for index in range(len(criteria))]
    raise DatasetValidationError("question type must be choice, noul, or score")


def validate_case(case: Dict[str, Any], seen_ids: set) -> int:
    """Validate one source record and return its number of decision targets."""
    case_id = case.get("id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise DatasetValidationError("id must be a non-empty string")
    if case_id in seen_ids:
        raise DatasetValidationError("id is duplicated: %s" % case_id)
    seen_ids.add(case_id)

    if not isinstance(case.get("state"), (str, dict, list)):
        raise DatasetValidationError("state must be text, a JSON object, or a JSON array")
    questions = case.get("questions")
    gold = case.get("gold")
    if not isinstance(questions, dict) or not questions:
        raise DatasetValidationError("questions must be a non-empty object")
    if not isinstance(gold, dict) or set(gold) != set(questions):
        raise DatasetValidationError("gold must contain exactly the same question ids as questions")

    for question_id, question in questions.items():
        if not isinstance(question_id, str) or not isinstance(question, dict):
            raise DatasetValidationError("questions must map string ids to objects")
        if not isinstance(question.get("instructions"), str):
            raise DatasetValidationError("question %r needs string instructions" % question_id)
        expected_keys = expected_probability_keys(question)
        gold_question = gold[question_id]
        if not isinstance(gold_question, dict) or not isinstance(gold_question.get("probabilities"), dict):
            raise DatasetValidationError("gold.%s needs a probabilities object" % question_id)
        probabilities = gold_question["probabilities"]
        if set(probabilities) != set(expected_keys):
            raise DatasetValidationError(
                "gold.%s probability keys must be exactly %s" % (question_id, expected_keys)
            )
        values = list(probabilities.values())
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise DatasetValidationError("gold.%s probabilities must be numeric" % question_id)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise DatasetValidationError("gold.%s probabilities must be finite and non-negative" % question_id)
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise DatasetValidationError("gold.%s probabilities must sum to 1.0" % question_id)
    return len(questions)


def read_cases(path: Path) -> Tuple[List[Dict[str, Any]], List[str], int]:
    """Read one JSONL file without writing output until every row is validated."""
    cases: List[Dict[str, Any]] = []
    errors: List[str] = []
    decisions = 0
    seen_ids: set = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            case = json.loads(raw_line)
            if not isinstance(case, dict):
                raise DatasetValidationError("each line must be a JSON object")
            decisions += validate_case(case, seen_ids)
            cases.append(case)
        except (DatasetValidationError, json.JSONDecodeError) as error:
            errors.append("line %d: %s" % (line_number, error))
    if not cases and not errors:
        errors.append("the file contains no JSONL records")
    return cases, errors, decisions


def notebook_row(case: Dict[str, Any]) -> Dict[str, str]:
    """Convert nested source JSON into the string fields expected by the upstream notebook."""
    return {
        "id": case["id"],
        "state": json.dumps(case["state"], ensure_ascii=False),
        "questions": json.dumps(case["questions"], ensure_ascii=False),
        "gold": json.dumps(case["gold"], ensure_ascii=False),
    }


def write_notebook_rows(cases: Iterable[Dict[str, Any]], path: Path) -> None:
    """Write already-validated rows in the upstream notebook's JSONL field shape."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for case in cases:
            output.write(json.dumps(notebook_row(case), ensure_ascii=False) + "\n")


def main() -> None:
    """Validate source JSONL and optionally export it for the official Kaggle notebook."""
    parser = argparse.ArgumentParser(description="Validate a Laya typed-decision JSONL dataset")
    parser.add_argument("dataset", type=Path, help="Nested source JSONL dataset")
    parser.add_argument("--output", type=Path, help="Optional notebook-compatible JSONL output")
    args = parser.parse_args()

    try:
        cases, errors, decisions = read_cases(args.dataset)
    except FileNotFoundError as error:
        raise SystemExit("dataset file does not exist: %s" % args.dataset) from error
    if errors:
        raise SystemExit("Dataset validation failed:\n" + "\n".join(errors))

    print("Valid: %d cases, %d decision targets" % (len(cases), decisions))
    if args.output is not None:
        write_notebook_rows(cases, args.output)
        print("Notebook-compatible JSONL: %s" % args.output)


if __name__ == "__main__":
    main()
