import json
from seq2seq_sar.utils import detect_serialized_output, ensure_human_readable_sar
from seq2seq_sar.serializer import example_target_text, serialize_account
from seq2seq_sar.data_generator import generate_transactions_for_account
from seq2seq_sar.aggregation import compute_aggregates
from seq2seq_sar.evaluation import rule_based_traceability_check


def test_detect_serialized_output_and_fallback():
    acc = "ACC_0001"
    txs = generate_transactions_for_account(acc, 2025, 12, 5, seed=1)
    agg = compute_aggregates(txs)
    src = serialize_account(acc, 2025, 12, txs, agg)

    # Simulate a serialized model output
    serialized_out = src  # reusing serialized source as a worst-case model output
    assert detect_serialized_output(serialized_out)

    human = ensure_human_readable_sar(serialized_out, acc, txs, agg)
    assert human == example_target_text(acc, txs, agg)

    # Traceability should pass for the deterministic template
    check = rule_based_traceability_check(src, human)
    assert check["ok"]


def test_preserve_human_readable_output():
    acc = "ACC_0002"
    txs = generate_transactions_for_account(acc, 2025, 12, 4, seed=2)
    agg = compute_aggregates(txs)
    src = serialize_account(acc, 2025, 12, txs, agg)

    human_in = example_target_text(acc, txs, agg)
    assert not detect_serialized_output(human_in)

    out = ensure_human_readable_sar(human_in, acc, txs, agg)
    assert out == human_in
    check = rule_based_traceability_check(src, out)
    assert check["ok"]
