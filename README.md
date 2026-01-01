# Table to Text — SAR Generation

Short, focused information for developers who want to run and contribute to the project.

## What this project does
Generates regulator-ready Suspicious Activity Reports (SARs) from structured transaction data using either deterministic templates or fine-tuned seq2seq models (T5/BART). Outputs are auditable and traceable back to source transactions.

## Quick start
1. Create and activate your environment (example):

   conda env create -f environment.yml -n table2text
   conda activate table2text

2. Install the project in editable mode:

   pip install -e .

3. Run a tiny end-to-end smoke experiment (programmatic):

   python -c "from seq2seq_sar.experiments import run_tiny_end_to_end; run_tiny_end_to_end(n_accounts=10, epochs=0, model='t5-small', seed=1, out_dir='outputs/example_run_notebook')"

4. Run tests:

   python -m pytest -q

## Layout & packaging
- Source package is under `src/seq2seq_sar/` (src layout). If a build environment cannot run editable installs, use `PYTHONPATH=src pytest ...` as a fallback.
- Notebooks live in `notebooks/`. `manual_run.ipynb` is cleaned to assume the package is installed (no automatic installer cell).

## Outputs
Generated artifacts (data, models, pipeline JSON, EDA) are written to `outputs/` (e.g., `outputs/example_run_notebook/` for the tiny experiment).

## Notes
- The repo is structured for local, deterministic runs — no external LLM APIs are used.
- If you want help initializing git, preparing commits, or adding a CI fallback (PYTHONPATH-based test step), tell me and I can prepare exact commands or a small workflow change.

---


If you'd like, I can add a short example command that runs a tiny end-to-end experiment and saves results to `outputs/example_run/` (train for 1 epoch on 100 synthetic accounts, infer 10 accounts, run trace checks and collate reports). Want me to add that? 
