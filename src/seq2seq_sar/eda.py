"""Exploratory Data Analysis for synthetic SAR data.
Generates transactions per config, computes aggregates, saves CSVs, serialized pairs,
and simple plots (amount histogram, tx type counts, country counts).
"""
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List

import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .data_generator import generate_monthly_for_accounts, generate_transactions_for_account
from .aggregation import compute_aggregates
from .serializer import serialize_account, example_target_text


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def generate_and_analyze(config_path: str = "config.yaml") -> dict:
    cfg = load_config(config_path)
    n_accounts = int(cfg.get("n_accounts", 50))
    prefix = cfg.get("account_prefix", "ACC_")
    year = int(cfg.get("year", 2025))
    month = int(cfg.get("month", 12))
    tx_range = tuple(cfg.get("tx_per_account_range", (5, 20)))
    seed = int(cfg.get("seed", 42))

    outputs = cfg.get("outputs", {})
    data_dir = Path(outputs.get("data_dir", "outputs/data"))
    eda_dir = Path(outputs.get("eda_dir", "outputs/eda"))

    ensure_dir(data_dir)
    ensure_dir(eda_dir)

    # Build account list
    account_ids = [f"{prefix}{i:04d}" for i in range(1, n_accounts + 1)]
    # Generate transactions
    all_txs = generate_monthly_for_accounts(account_ids, year, month, tx_range=tx_range, seed=seed)
    df = pd.DataFrame(all_txs)
    txs_csv = data_dir / "transactions.csv"
    df.to_csv(txs_csv, index=False)

    # Aggregates per account
    aggs = []
    pairs = []
    for acc in account_ids:
        acc_tx = df[df["account_id"] == acc].to_dict(orient="records")
        agg = compute_aggregates(acc_tx)
        aggs.append({"account_id": acc, **agg.__dict__})
        src = serialize_account(acc, year, month, acc_tx, agg)
        tgt = example_target_text(acc, acc_tx, agg)
        pairs.append({"account_id": acc, "input": src, "target": tgt})

    pd.DataFrame(aggs).to_csv(data_dir / "aggregates.csv", index=False)
    if cfg.get("save_pairs", True):
        pairs_df = pd.DataFrame(pairs)
        pairs_df.to_csv(data_dir / "pairs.csv", index=False)
        # also save a JSON sample
        sample_n = min(cfg.get("pairs_sample_limit", 100), len(pairs))
        with open(data_dir / "pairs_sample.json", "w", encoding="utf-8") as f:
            json.dump(pairs[:sample_n], f, indent=2)

    # EDA plots
    eda_cfg = cfg.get("eda", {})
    # Amount histogram
    if eda_cfg.get("plot_amount_hist", True):
        plt.figure(figsize=(6, 4))
        sns.histplot(df["amount"].dropna(), bins=50, kde=False)
        plt.title("Transaction Amounts")
        plt.savefig(eda_dir / "amount_hist.png")
        plt.close()

    # Tx types
    if eda_cfg.get("plot_tx_types", True):
        plt.figure(figsize=(6, 4))
        sns.countplot(y=df["transaction_type"])
        plt.title("Transaction Types")
        plt.savefig(eda_dir / "tx_types.png")
        plt.close()

    # Country counts
    if eda_cfg.get("plot_country_counts", True):
        plt.figure(figsize=(6, 4))
        sns.countplot(y=df["country"])
        plt.title("Country distribution")
        plt.savefig(eda_dir / "country_counts.png")
        plt.close()

    # Summary stats
    summary = {
        "n_transactions": int(len(df)),
        "n_accounts": int(df["account_id"].nunique()),
        "amount_mean": float(df["amount"].mean()),
        "amount_median": float(df["amount"].median()),
        "alert_rate": float(df["alert"].mean()),
    }
    with open(eda_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {"transactions_csv": str(txs_csv), "pairs_csv": str(data_dir / "pairs.csv"), "eda_dir": str(eda_dir), "summary": summary}


if __name__ == "__main__":
    out = generate_and_analyze()
    print(out)
