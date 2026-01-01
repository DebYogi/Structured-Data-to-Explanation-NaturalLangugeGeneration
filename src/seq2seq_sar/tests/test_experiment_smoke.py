import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from seq2seq_sar.experiments import run_tiny_end_to_end
from seq2seq_sar import train as train_module
from seq2seq_sar.utils import detect_serialized_output


def test_tiny_experiment_smoke(tmp_path, monkeypatch):
    out_dir = tmp_path / "example_run"

    # Monkeypatch the heavy training step to be a lightweight stub that creates a dummy model dir
    def fake_train_main(args: SimpleNamespace):
        model_dir = Path(args.output_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text(json.dumps({"dummy": True}))
        (model_dir / "pytorch_model.bin").write_text("")

    monkeypatch.setattr(train_module, "main", fake_train_main)

    # Monkeypatch generate_sar to avoid loading HF models; return deterministic template instead
    from seq2seq_sar.data_generator import generate_transactions_for_account
    from seq2seq_sar.aggregation import compute_aggregates
    from seq2seq_sar.serializer import example_target_text

    def fake_generate_sar(model_dir, acc, year, month, n_tx=10, seed=42):
        txs = generate_transactions_for_account(acc, year, month, n_tx, seed=seed)
        agg = compute_aggregates(txs)
        return example_target_text(acc, txs, agg)

    monkeypatch.setattr("seq2seq_sar.experiments.generate_sar", fake_generate_sar)

    res = run_tiny_end_to_end(n_accounts=5, epochs=0, model="t5-small", seed=1, out_dir=str(out_dir), use_accelerate=False)
    outd = Path(res)

    # Check summary exists
    summary = outd / "summary.json"
    assert summary.exists()
    s = json.loads(summary.read_text())
    assert s["n_accounts"] == 5

    # Check pipeline outputs: one sar per sampled account
    pipeline = outd / "pipeline"
    jsons = list(pipeline.glob("sar_*.json"))
    assert len(jsons) >= 1

    for j in jsons:
        rec = json.loads(j.read_text())
        assert "sar_text" in rec
        # sar_text should be human-readable (not serialized tokens)
        assert not detect_serialized_output(rec["sar_text"]) , f"Serialized output detected in {j}"
        assert "traceability" in rec
        assert rec["traceability"]["ok"] is True
