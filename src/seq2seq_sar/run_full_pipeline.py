"""Run small end-to-end pipeline: inference for sample accounts and traceability checks.
Writes outputs to outputs/pipeline/ and prints summary.
"""
import json
from pathlib import Path
from seq2seq_sar.infer import generate_sar
from seq2seq_sar.data_generator import generate_transactions_for_account
from seq2seq_sar.aggregation import compute_aggregates
from seq2seq_sar.serializer import serialize_account, example_target_text
from seq2seq_sar.evaluation import rule_based_traceability_check

OUT_DIR = Path("outputs/pipeline")
OUT_DIR.mkdir(parents=True, exist_ok=True)

accounts = [("ACC_0001", 2), ("ACC_0002", 3), ("ACC_0003", 4)]
results = []
for acc, seed in accounts:
    txs = generate_transactions_for_account(acc, 2025, 12, 10, seed=seed)
    agg = compute_aggregates(txs)
    src = serialize_account(acc, 2025, 12, txs, agg)
    out = generate_sar("outputs/sar_model_full", acc, 2025, 12, n_tx=10, seed=seed)
    # Normalize output: prefer human-readable deterministic template when model output looks serialized
    from .utils import ensure_human_readable_sar
    out = ensure_human_readable_sar(out, acc, txs, agg)

    check = rule_based_traceability_check(src, out)
    rec = {
        "account_id": acc,
        "source": src,
        "sar_text": out,
        "traceability": check,
    }
    results.append(rec)
    (OUT_DIR / f"sar_{acc}_2025-12.json").write_text(json.dumps(rec, indent=2))

# Print summary
for r in results:
    print(f"Account {r['account_id']}: trace ok={r['traceability']['ok']}, issues={r['traceability']['issues']}")

print(f"Wrote {len(results)} SAR JSON(s) to {OUT_DIR}")
