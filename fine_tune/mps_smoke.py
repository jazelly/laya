"""Run one real forward, backward, and optimizer step on Apple GPU without saving weights."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict

import torch
import torch.nn.functional as functional

import laya
from laya.common import QTYPES, build_sequence


def main() -> None:
    """Prove the core Laya gradient path works on the requested local device."""
    parser = argparse.ArgumentParser(description="Laya MPS fine-tuning smoke test")
    parser.add_argument("--model", default="convaiinnovations/laya")
    parser.add_argument("--device", default="mps", choices=("mps", "cpu"))
    args = parser.parse_args()

    agent = laya.load(args.model, device=args.device)
    if str(agent.device) != args.device:
        raise SystemExit("requested %s but Laya loaded on %s" % (args.device, agent.device))

    question: Dict[str, Any] = {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "Invoices, payments, and refunds",
            "technical": "Software bugs and outages",
            "sales": "Pricing and new purchases",
        },
    }
    internal_question = {
        "t": question["type"],
        "ins": question["instructions"],
        "crit": question["criteria"],
    }
    token_ids, markers = build_sequence(
        agent.tok,
        {"message": "I was charged twice for invoice 4411. Please refund the duplicate today."},
        internal_question,
        agent.cfg["max_len"],
        agent.cfg["head_max_len"],
    )
    device = torch.device(args.device)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    marker_pos = torch.tensor([markers], dtype=torch.long, device=device)
    marker_mask = torch.ones((1, len(markers)), dtype=torch.bool, device=device)
    question_type = torch.tensor([QTYPES["choice"]], dtype=torch.long, device=device)
    target = torch.tensor([0], dtype=torch.long, device=device)

    agent.model.train()
    optimizer = torch.optim.SGD(agent.model.parameters(), lr=1e-7)
    started = time.perf_counter()
    logits, action = agent.model(input_ids, attention_mask, marker_pos, marker_mask, question_type)
    loss = functional.cross_entropy(logits[:, : len(markers)].float(), target) + action.sum() * 0
    loss.backward()
    optimizer.step()

    print(
        json.dumps(
            {
                "device": str(agent.device),
                "loss": round(float(loss.detach().float().cpu().item()), 6),
                "backward_and_step_ms": round((time.perf_counter() - started) * 1000, 1),
                "saved_checkpoint": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
