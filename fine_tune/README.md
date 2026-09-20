# Fine-tuning Laya

This fork separates two claims that must not be conflated:

1. `mps_smoke.py` proves one real Laya forward pass, backward pass, and optimizer step on this
   Mac's Apple GPU. It does **not** claim that the whole upstream RLCD training recipe is
   validated on MPS.
2. The upstream full training recipe remains
   [`notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`](../notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb).
   It is explicitly CUDA + 2×T4 + DDP. Use it as the authoritative full-training path until a
   complete MPS run has its own evidence.

## Dataset contract

One JSONL line is one **case**:

```json
{
  "id": "stable-case-id",
  "state": {"message": "the input your application will receive"},
  "questions": {"one typed decision schema": {}},
  "gold": {"one probability distribution for every question": {}}
}
```

Use [`dataset_template.jsonl`](dataset_template.jsonl) as the exact nested source shape. The
validator requires all possible outcomes to be present in each `gold.probabilities` object:

| Type | Required probability keys |
| --- | --- |
| `choice` | Exactly the option labels in `criteria` |
| `noul` | `false`, `true` |
| `score` | `0` through `N-1`, matching the ordered criteria list |

The values must be non-negative, finite, and sum to 1.0. A hard human label is represented with
one value of `1.0`; a genuine adjudicated uncertainty can be represented as a soft distribution.
Do not turn the base model's own output into gold data.

Split by the real-world unit that can leak context (customer, ticket thread, document, or event),
not by individual questions from the same state. Keep a held-out test set before training. Remove
credentials, personal identifiers, and private content that you are not authorized to put into a
training dataset.

For a smoke dataset, twenty carefully labeled cases can prove the pipeline. For the first useful
domain experiment, aim for hundreds of cases with a stable schema and a held-out evaluation set.
The upstream reference notebook preprocesses 1,200 cases into roughly 30,000 decision sequences;
that is a reference workload, not an asserted minimum for every domain.

## Validate and export local data

Keep source records nested and readable, then validate them before any training run:

```bash
.venv/bin/python -m fine_tune.validate_dataset fine_tune/dataset_template.jsonl
```

The upstream notebook calls `json.loads(row["state"])` (and likewise for questions and gold), so
its dataset rows use JSON **strings**. Export a compatible file only after validation:

```bash
.venv/bin/python -m fine_tune.validate_dataset \
  data/train.source.jsonl \
  --output data/train.jsonl
```

In a copy of the upstream Kaggle notebook, replace its training dataset load with the exported
file:

```python
ds_train = load_dataset(
    "json",
    data_files="/kaggle/input/your-dataset/train.jsonl",
    split="train",
)
```

Do the same for a held-out `test.jsonl` in the notebook's evaluation cell. Keep the dataset local
or in your own approved storage; publishing a checkpoint or dataset is deliberately not automated
by this fork.

## Apple GPU training-path smoke test

After the standard editable install, run:

```bash
.venv/bin/python -m fine_tune.mps_smoke --device mps
```

It loads the official English checkpoint, computes a real loss, backpropagates through the model,
and calls one SGD optimizer step in memory. It does not write a checkpoint or replace model files.
If it passes, the machine's MPS forward/backward core is evidenced; it still does not substitute
for the upstream CUDA DDP, RLCD, temperature-calibration, and held-out-evaluation loop.
