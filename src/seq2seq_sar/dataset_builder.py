"""Build training dataset of (input_text, target_text) pairs and provide PyTorch Dataset.
"""
from typing import List, Tuple
from torch.utils.data import Dataset

from .data_generator import generate_monthly_for_accounts
from .aggregation import compute_aggregates
from .serializer import serialize_account, example_target_text

import pandas as pd

class SARPairDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], tokenizer, max_length=512):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_enc = self.tokenizer(src, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        tgt_enc = self.tokenizer(tgt, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {"input_ids": src_enc['input_ids'].squeeze(), "attention_mask": src_enc['attention_mask'].squeeze(), "labels": tgt_enc['input_ids'].squeeze()}


def build_pairs(account_ids: List[str], year: int, month: int, seed: int = 0) -> List[Tuple[str, str]]:
    txs = generate_monthly_for_accounts(account_ids, year, month, seed=seed)
    # group by account
    by_acc = {}
    for t in txs:
        by_acc.setdefault(t['account_id'], []).append(t)
    pairs = []
    for acc, transactions in by_acc.items():
        agg = compute_aggregates(transactions)
        src = serialize_account(acc, year, month, transactions, agg)
        tgt = example_target_text(acc, transactions, agg)
        pairs.append((src, tgt))
    return pairs


def build_pairs_and_save(account_ids: List[str], year: int, month: int, seed: int = 0, out_dir: str = "outputs/data", run_eda: bool = False, config_path: str = None) -> str:
    """Build pairs and save to CSV. Optionally run EDA which will save additional artifacts.

    Returns path to pairs CSV file.
    """
    pairs = build_pairs(account_ids, year, month, seed=seed)
    from pathlib import Path
    outd = Path(out_dir)
    outd.mkdir(parents=True, exist_ok=True)
    pairs_df = pd.DataFrame([{"account_id": a, "input": s, "target": t} for (s, t), a in zip(pairs, [p[0].split('|')[0].split('=')[1].strip() for p in pairs])])
    pairs_csv = outd / "pairs.csv"
    pairs_df.to_csv(pairs_csv, index=False)

    # Optionally run EDA to generate plots and summary (uses config file if provided)
    if run_eda:
        # import here to avoid circular imports
        from .eda import generate_and_analyze
        if config_path:
            generate_and_analyze(config_path)
        else:
            # build a temporary config that points outputs to this out_dir
            temp_cfg = {
                "n_accounts": len(account_ids),
                "account_prefix": "",
                "year": year,
                "month": month,
                "tx_per_account_range": [len(pairs) // max(1, len(account_ids)), len(pairs)],
                "seed": seed,
                "outputs": {"data_dir": str(outd), "eda_dir": str(Path(outd) / "eda")},
                "save_pairs": True,
                "pairs_sample_limit": min(100, len(pairs)),
                "eda": {"plot_amount_hist": True, "plot_tx_types": True, "plot_country_counts": True},
            }
            import tempfile
            import yaml
            tf = Path(tempfile.gettempdir()) / "seq2seq_sar_temp_config.yaml"
            tf.write_text(yaml.safe_dump(temp_cfg))
            generate_and_analyze(str(tf))
    return str(pairs_csv)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-accounts", type=int, default=10)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--month", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="outputs/data")
    parser.add_argument("--run-eda", action="store_true", dest="run_eda", help="Run EDA after building pairs")
    parser.add_argument("--config", type=str, default=None, help="Optional config.yaml to pass to EDA")
    args = parser.parse_args()
    account_ids = [f"ACC_{i:04d}" for i in range(1, args.n_accounts + 1)]
    p = build_pairs_and_save(account_ids, args.year, args.month, seed=args.seed, out_dir=args.out_dir, run_eda=args.run_eda, config_path=args.config)
    print(f"Wrote pairs to {p}")
