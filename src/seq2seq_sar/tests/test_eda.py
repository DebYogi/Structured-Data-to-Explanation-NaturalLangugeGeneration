from seq2seq_sar.eda import generate_and_analyze


def test_generate_and_analyze_creates_outputs(tmp_path, monkeypatch):
    # Use temp config to direct outputs to tmp_path
    cfg = {
        "n_accounts": 5,
        "account_prefix": "ACC_",
        "year": 2025,
        "month": 12,
        "tx_per_account_range": [3, 4],
        "seed": 1,
        "outputs": {"data_dir": str(tmp_path / "data"), "eda_dir": str(tmp_path / "eda")},
        "save_pairs": True,
        "pairs_sample_limit": 10,
        "eda": {"plot_amount_hist": False, "plot_tx_types": False, "plot_country_counts": False},
    }
    # write temp config
    cfg_path = tmp_path / "config.yaml"
    import yaml
    cfg_path.write_text(yaml.safe_dump(cfg))

    res = generate_and_analyze(str(cfg_path))
    assert 'transactions.csv' in res['transactions_csv']
    assert 'pairs.csv' in res['pairs_csv']
    assert tmp_path.joinpath('data','transactions.csv').exists()
    assert tmp_path.joinpath('data','pairs.csv').exists()
    assert tmp_path.joinpath('eda','summary.json').exists()
