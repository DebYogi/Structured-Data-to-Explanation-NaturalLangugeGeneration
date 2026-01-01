"""Inference script to generate SARs from a trained seq2seq checkpoint.
Uses deterministic generation (set seed, greedy or beam with fixed seeds).
"""
import argparse
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from .data_generator import generate_transactions_for_account
from .aggregation import compute_aggregates
from .serializer import serialize_account


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_sar(model_name_or_path: str, account_id: str, year: int, month: int, n_tx: int = 10, seed: int = 42, max_length: int = 256):
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    txs = generate_transactions_for_account(account_id, year, month, n_tx, seed=seed)
    agg = compute_aggregates(txs)
    src = serialize_account(account_id, year, month, txs, agg)

    inputs = tokenizer(src, return_tensors='pt', truncation=True)
    # deterministic greedy decoding
    outputs = model.generate(**inputs, max_length=max_length, num_beams=1, do_sample=False)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="t5-small")
    parser.add_argument("--account", type=str, default="ACC_1001")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--month", type=int, default=12)
    parser.add_argument("--n-tx", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(generate_sar(args.checkpoint, args.account, args.year, args.month, args.n_tx, args.seed))
