from seq2seq_sar.dataset_builder import build_pairs_and_save
import yaml
from pathlib import Path


def test_build_pairs_and_run_eda(tmp_path, monkeypatch):
    account_ids = [f"ACC_{i:04d}" for i in range(1, 6)]
    out_dir = tmp_path / "data"
    out_dir.mkdir()

    # Create a small config file
    cfg = {
        "n_accounts": 5,
        "account_prefix": "ACC_",
        "year": 2025,
        "month": 12,
        "tx_per_account_range": [3, 4],
        "seed": 1,
        "outputs": {"data_dir": str(out_dir), "eda_dir": str(tmp_path / "eda")},
        "save_pairs": True,
        "pairs_sample_limit": 10,
        "eda": {"plot_amount_hist": False, "plot_tx_types": False, "plot_country_counts": False},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    pairs_csv = build_pairs_and_save(account_ids, 2025, 12, seed=1, out_dir=str(out_dir), run_eda=True, config_path=str(cfg_path))
    assert Path(pairs_csv).exists()
    assert (out_dir / "pairs.csv").exists()
    assert (tmp_path / "eda" / "summary.json").exists()
