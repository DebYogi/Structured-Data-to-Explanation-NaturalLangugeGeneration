# Table to Text — SAR Generation (Seq2Seq + Templates)

A concise but comprehensive developer guide with configuration, detailed run steps, troubleshooting tips, and CI recommendations.

## What this repo does
Generates regulator-ready Suspicious Activity Reports (SARs) from synthetic transaction data. It supports:
- Deterministic, template-based pipeline (auditable and rule-guarded)
- Seq2Seq fine-tuning (T5/BART) to translate structured input into human-readable SAR narratives

All outputs are traceable to source transactions and the code includes rule-based checks to validate traceability.

---

## Prerequisites
- Python >= 3.10 (Conda recommended)
- git
- (Optional) CUDA + GPU for training large models — not required for smoke tests

Quick setup example:

```bash
conda env create -f environment.yml -n table2text
conda activate table2text
```

---

## Install (editable)
From project root:

```bash
pip install -e .
```

If editable install fails in your build environment, use the `PYTHONPATH=src` fallback (see Testing section).

---

## Configuration
Primary configuration: `seq2seq_sar/config.yaml` plus CLI flags. Key settings:
- Data & EDA: `n_accounts`, `year`, `month`, `tx_per_account_range`, `seed`, `outputs.data_dir`, `outputs.eda_dir`, `eda.plot_*`
- Training CLI: `--model` (e.g., `t5-small`), `--epochs`, `--batch-size`, `--lr`, `--use-accelerate`
- Inference CLI: `--checkpoint`, `--account`, `--year`, `--month`, `--n-tx`, `--seed`

See individual modules for more options.

---

## Run examples
Build data + EDA:

```bash
python -m seq2seq_sar.dataset_builder --n-accounts 10 --run-eda --out_dir outputs/example_run_notebook/data --seed 1
```

Train (tiny smoke test):

```bash
python -m seq2seq_sar.train --model t5-small --epochs 0 --output-dir outputs/example_run_notebook/model
```

Run pipeline / inference:

```bash
python -m seq2seq_sar.run_full_pipeline --out_dir outputs/example_run_notebook
```

Tiny E2E helper:

```bash
python -c "from seq2seq_sar.experiments import run_tiny_end_to_end; run_tiny_end_to_end(n_accounts=10, epochs=0, model='t5-small', seed=1, out_dir='outputs/example_run_notebook')"
```

Inspect outputs:
- Models: `outputs/example_run_notebook/model/`
- SAR JSONs: `outputs/example_run_notebook/pipeline/`
- EDA: `outputs/example_run_notebook/eda/`

---

## Notebooks
- `notebooks/manual_run.ipynb` — a tiny E2E demo. The notebook assumes the project is installed in your environment and *does not* auto-install. If you prefer not to install, run the notebook with `PYTHONPATH=src` or temporarily uncomment the commented `sys.path` lines in the notebook.

---

## Testing
Run tests locally:

```bash
python -m pytest -q
```

If editable installs are not possible in CI/build environments, use this fallback:

```bash
# posix
PYTHONPATH=src python -m pytest -q
# Windows cmd
set PYTHONPATH=src&& python -m pytest -q
```

Quick programmatic smoke test:

```bash
python tools/run_smoke_tests.py
```

---

## Troubleshooting
- ModuleNotFoundError: No module named 'seq2seq_sar'
  - Fix: `pip install -e .` in your active environment (restart interpreters/kernels), or use `PYTHONPATH=src`.

- `pip install -e .` errors with `egg_base 'src' does not exist`
  - Fix: either move the package into `src/` or update `pyproject.toml` to find packages in `'.'`.

- Editable install fails due to Python mismatch during isolated build
  - Fix: ensure your build environment uses a Python compatible with `requires-python` in `pyproject.toml` (>= 3.10).

---

## CI recommendation
Add a small step for runners that cannot perform editable installs, e.g. GitHub Actions step:

```yaml
- name: Run tests (fallback PYTHONPATH)
  run: |
    set PYTHONPATH=src&& python -m pytest -q
  shell: cmd
```

Optionally add a dedicated smoke job that runs the tiny end-to-end helper to ensure end-to-end expectations remain stable.

---

