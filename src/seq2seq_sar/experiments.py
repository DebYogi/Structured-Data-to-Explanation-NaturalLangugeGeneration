"""Run a tiny end-to-end experiment: data build + EDA, small training, inference, and traceability checks.

This is intended as a reproducible smoke test / example that saves results under `outputs/example_run/`.
"""
import os
from pathlib import Path
import json
from argparse import ArgumentParser, Namespace

from .dataset_builder import build_pairs_and_save
from . import train as train_module
from .infer import generate_sar
from .data_generator import generate_transactions_for_account
from .aggregation import compute_aggregates
from .evaluation import rule_based_traceability_check


def run_tiny_end_to_end(n_accounts: int = 50, epochs: int = 1, model: str = "t5-small", seed: int = 42, out_dir: str = "outputs/example_run", use_accelerate: bool = False):
    outd = Path(out_dir)
    data_dir = outd / "data"
    model_dir = outd / "model"
    pipeline_dir = outd / "pipeline"
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    # Build dataset and run EDA
    account_ids = [f"ACC_{i:04d}" for i in range(1, n_accounts + 1)]
    print(f"Building dataset for {n_accounts} accounts (seed={seed}) and running EDA; outputs -> {data_dir}")
    build_pairs_and_save(account_ids, 2025, 12, seed=seed, out_dir=str(data_dir), run_eda=True)

    # Train a small model (calls train.main)
    print(f"Training model {model} for {epochs} epoch(s); outputs -> {model_dir}")
    args = Namespace(
        model=model,
        n_accounts=n_accounts,
        year=2025,
        month=12,
        epochs=epochs,
        batch_size=None,
        max_len=256,
        output_dir=str(model_dir),
        lr=5e-5,
        seed=seed,
        device="auto",
        fp16=False,
        gradient_accumulation_steps=1,
        use_accelerate=use_accelerate,
    )
    train_module.main(args)

    # Inference for a small set of accounts
    sample_accounts = account_ids[:5]
    results = []
    from .serializer import serialize_account
    from .utils import ensure_human_readable_sar

    for acc in sample_accounts:
        txs = generate_transactions_for_account(acc, 2025, 12, 10, seed=seed)
        agg = compute_aggregates(txs)
        src = serialize_account(acc, 2025, 12, txs, agg)
        out = generate_sar(str(model_dir), acc, 2025, 12, n_tx=10, seed=seed)
        out = ensure_human_readable_sar(out, acc, txs, agg)
        # Traceability uses source + output
        check = rule_based_traceability_check(src, out)
        rec = {
            "account_id": acc,
            "source": src,
            "sar_text": out,
            "traceability": check,
        }
        results.append(rec)
        (pipeline_dir / f"sar_{acc}_2025-12.json").write_text(json.dumps(rec, indent=2))

    # Write an index summary
    (outd / "summary.json").write_text(json.dumps({"n_accounts": n_accounts, "model": model, "results": results}, indent=2))
    print(f"Experiment complete. Wrote model -> {model_dir}; pipeline outputs -> {pipeline_dir}; summary -> {outd / 'summary.json'}")
    return str(outd)


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--n-accounts", type=int, default=50)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--model", type=str, default="t5-small")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="outputs/example_run")
    p.add_argument("--use-accelerate", action="store_true", dest="use_accelerate")
    args = p.parse_args()
    run_tiny_end_to_end(n_accounts=args.n_accounts, epochs=args.epochs, model=args.model, seed=args.seed, out_dir=args.out_dir, use_accelerate=args.use_accelerate)