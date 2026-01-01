"""Deterministic aggregation functions used in input construction and validation."""
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Aggregates:
    total_debit: float
    total_credit: float
    num_debit: int
    num_credit: int
    unique_creditors: int
    unique_debtors: int
    tx_types: List[str]


def compute_aggregates(transactions: List[Dict]) -> Aggregates:
    debit_amount = 0.0
    credit_amount = 0.0
    num_debit = 0
    num_credit = 0
    creditors = set()
    debtors = set()
    types = set()
    for tx in transactions:
        amt = float(tx.get("amount", 0.0))
        if tx.get("debit_or_credit") == "debit":
            debit_amount += amt
            num_debit += 1
            if tx.get("counterparty_role") == "creditor":
                creditors.add(tx.get("counterparty_id"))
        else:
            credit_amount += amt
            num_credit += 1
            if tx.get("counterparty_role") == "debtor":
                debtors.add(tx.get("counterparty_id"))
        types.add(tx.get("transaction_type"))
    return Aggregates(
        total_debit=round(debit_amount, 2),
        total_credit=round(credit_amount, 2),
        num_debit=num_debit,
        num_credit=num_credit,
        unique_creditors=len(creditors),
        unique_debtors=len(debtors),
        tx_types=sorted(list(types)),
    )
