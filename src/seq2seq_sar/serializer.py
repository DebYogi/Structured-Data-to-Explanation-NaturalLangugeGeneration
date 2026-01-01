"""Serialize structured transaction data and aggregates into a deterministic text input
suitable as source for seq2seq fine-tuning.
"""
from typing import List, Dict
from .aggregation import Aggregates


def serialize_account(account_id: str, year: int, month: int, transactions: List[Dict], aggregates: Aggregates) -> str:
    parts = []
    parts.append(f"ACCOUNT_ID={account_id} | MONTH={year:04d}-{month:02d}")
    for i, tx in enumerate(transactions, start=1):
        tx_parts = [f"TXN_{i}:"]
        tx_parts.append(f"DATE={tx.get('transaction_date')}")
        tx_parts.append(tx.get('debit_or_credit'))
        tx_parts.append(f"{tx.get('currency')} {tx.get('amount')}")
        tx_parts.append(tx.get('transaction_type'))
        # counterparty
        role = tx.get('counterparty_role')
        cp = tx.get('counterparty_id')
        tx_parts.append(f"{role.upper()}={cp}")
        if tx.get('alert') == 1:
            tx_parts.append(f"ALERT=1")
            if tx.get('alert_reason'):
                tx_parts.append(f"REASON={tx.get('alert_reason')}")
        parts.append(
            ", ".join(tx_parts)
        )
    ag = aggregates
    parts.append("AGGREGATES:")
    parts.append(f"TOTAL_DEBIT={ag.total_debit} | TOTAL_CREDIT={ag.total_credit}")
    parts.append(f"DEBIT_COUNT={ag.num_debit} | CREDIT_COUNT={ag.num_credit}")
    parts.append(f"UNIQUE_CREDITORS={ag.unique_creditors} | UNIQUE_DEBTORS={ag.unique_debtors}")
    parts.append(f"TXN_TYPES={','.join(ag.tx_types)}")
    return " | ".join(parts)


def example_target_text(account_id: str, transactions: List[Dict], aggregates: Aggregates) -> str:
    # Deterministic template for generating target SAR text for synthetic training
    lines = [f"SUSPICIOUS ACTIVITY REPORT\n\nAccount ID: {account_id}\n\nPART 1: TRANSACTION DETAILS"]
    for tx in transactions:
        date = tx.get('transaction_date')
        dc = tx.get('debit_or_credit')
        amt = f"{tx.get('currency')} {tx.get('amount'):,.2f}"
        tx_type = tx.get('transaction_type')
        role = tx.get('counterparty_role')
        cp = tx.get('counterparty_id')
        reason = f" Alert reason: {tx.get('alert_reason')}" if tx.get('alert') == 1 else ""
        lines.append(f"On {date}, a {dc} of {amt} was executed via {tx_type}. The counterparty was a {role} (id: {cp}).{reason}")
    lines.append("\nPART 2: ACCOUNT SUMMARY")
    lines.append(
        f"During the reporting period, total debits amounted to {aggregates.total_debit:,.2f} and total credits amounted to {aggregates.total_credit:,.2f}. "
        f"There were {aggregates.num_debit} debit transactions and {aggregates.num_credit} credit transactions. "
        f"Unique creditors: {aggregates.unique_creditors}; unique debtors: {aggregates.unique_debtors}. Transactions observed: {', '.join(aggregates.tx_types)}."
    )
    return "\n".join(lines)
