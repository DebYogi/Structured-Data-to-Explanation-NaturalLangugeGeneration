"""Deterministic synthetic monthly transaction generator for seq2seq training.
Generates transactions with schema required by the SAR task.
"""
from __future__ import annotations

import uuid
import random
from datetime import datetime, timedelta
from typing import List, Dict

import numpy as np
import pandas as pd

TX_TYPES = ["wire", "cash", "ACH", "card", "crypto"]
COUNTRY_CODES = ["US", "GB", "NG", "IR", "PK"]
CURRENCIES = ["USD", "EUR", "GBP"]


def _random_date_in_month(year: int, month: int, rng: random.Random) -> str:
    start = datetime(year, month, 1)
    if month == 12:
        end = datetime(year + 1, 1, 1)
    else:
        end = datetime(year, month + 1, 1)
    delta = (end - start).days
    day = rng.randrange(delta)
    return (start + timedelta(days=day)).date().isoformat()


def generate_transactions_for_account(account_id: str, year: int, month: int, n_tx: int, seed: int = 0) -> List[Dict]:
    rng = random.Random(seed + abs(hash(account_id)) % (2 ** 16))
    np_rng = np.random.default_rng(seed + abs(hash(account_id)) % (2 ** 16))
    rows = []
    for i in range(n_tx):
        tx_id = str(uuid.uuid4())
        tx_date = _random_date_in_month(year, month, rng)
        amount = float(abs(np_rng.normal(loc=2000, scale=4000))) + 1.0
        tx_type = rng.choice(TX_TYPES)
        debit_or_credit = rng.choice(["debit", "credit"])
        counterparty_id = "CP_" + str(rng.randrange(1000, 99999))
        counterparty_role = rng.choice(["creditor", "debtor"]) if debit_or_credit == "debit" else rng.choice(["creditor", "debtor"])
        country = rng.choice(COUNTRY_CODES)
        currency = rng.choice(CURRENCIES)
        alert = 1 if (amount > 10000 or country in ("NG", "IR")) else 0
        alert_reason = []
        if amount > 10000:
            alert_reason.append("high_value")
        if country in ("NG", "IR"):
            alert_reason.append("high_risk_country")
        rows.append(
            {
                "account_id": account_id,
                "transaction_id": tx_id,
                "transaction_date": tx_date,
                "amount": round(amount, 2),
                "currency": currency,
                "debit_or_credit": debit_or_credit,
                "transaction_type": tx_type,
                "counterparty_id": counterparty_id,
                "counterparty_role": counterparty_role,
                "country": country,
                "alert": alert,
                "alert_reason": ",".join(alert_reason),
            }
        )
    # chronological
    rows = sorted(rows, key=lambda r: r["transaction_date"])
    return rows


def generate_monthly_for_accounts(account_ids: List[str], year: int, month: int, tx_range=(5, 20), seed: int = 0) -> List[Dict]:
    out = []
    rng = random.Random(seed)
    for i, acc in enumerate(account_ids):
        n_tx = rng.randrange(tx_range[0], tx_range[1] + 1)
        out.extend(generate_transactions_for_account(acc, year, month, n_tx, seed + i))
    return out


if __name__ == "__main__":
    print(generate_transactions_for_account("ACC_1001", 2025, 12, 5, seed=123))
