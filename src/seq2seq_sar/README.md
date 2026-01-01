# Seq2Seq SAR Generator (fine-tuning)

Overview
--------
This module fine-tunes a seq2seq Transformer (T5/BART) to convert serialized structured
transaction data into regulator-ready SAR narratives.

Key constraints:
- No LLM APIs or agent frameworks
- Deterministic, auditable outputs
- Aggregates computed programmatically (not by model)

Quickstart
----------
1. Install dependencies: `pip install -r requirements.txt` (see file)
2. Create training data: `python -m seq2seq_sar.dataset_builder --n-accounts 100 --seed 42`

Training (CPU/GPU)
------------------
- Default (Trainer):
  `python -m seq2seq_sar.train --model t5-small --n-accounts 50 --epochs 1`

- Use `accelerate` (recommended for multi-GPU and fp16):
  1. Install accelerate: `conda run -n table2text pip install accelerate`
  2. Run with the flag to use `Accelerator` internally:
     `conda run -n table2text python -m seq2seq_sar.train --model t5-small --n-accounts 50 --epochs 1 --use-accelerate`
  - The script will automatically enable fp16 when a CUDA GPU is present unless `--fp16 False` is passed.
  - For advanced multi-node/multi-GPU launch, prefer `accelerate launch`:
     `accelerate launch --config_file my_config.yaml python -m seq2seq_sar.train --model t5-small --use-accelerate`

4. Inference: `python -m seq2seq_sar.infer --checkpoint path/to/checkpoint --account ACC_1001 --year 2025 --month 12`

Example end-to-end tiny experiment
----------------------------------

Run a reproducible tiny experiment that builds data, runs EDA, trains a small model for 1 epoch, runs inference on a few accounts, and writes reports to `outputs/example_run/`:

`python -m seq2seq_sar.experiments --n-accounts 50 --epochs 1 --model t5-small --out-dir outputs/example_run`

Use `--use-accelerate` if you have `accelerate` configured for your GPU(s). The experiment is intended as a smoke test / demo and will produce a small model in `outputs/example_run/model` and SAR JSONs in `outputs/example_run/pipeline/`.

Notebooks
---------
- `notebooks/manual_run.ipynb` — run a small end-to-end experiment interactively and inspect the outputs (summary, pipeline SAR JSONs, EDA plots).
- `notebooks/update_workflows.ipynb` — example for programmatically writing workflow YAML and validating it locally.


Modules
-------
- data_generator.py
- aggregation.py
- serializer.py
- dataset_builder.py
- train.py
- infer.py
- evaluation.py

Compliance
----------
Follow the project compliance rules: no inferred facts or speculation. Evaluation includes rule-based checks for traceability.

CI smoke test
-------------
A GitHub Actions workflow (`.github/workflows/smoke_tests.yml`) runs on push and PRs to `main` using **Python 3.11** and executes the test suite as a quick smoke check. To run the smoke test locally (fast, with heavy training steps stubbed by the test):

`python -m pytest seq2seq_sar/tests/test_experiment_smoke.py -q`

