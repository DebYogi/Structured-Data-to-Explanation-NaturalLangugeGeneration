def test_imports():
    import seq2seq_sar.serializer as s
    assert hasattr(s, "serialize_account")


def test_example_target():
    from seq2seq_sar.serializer import example_target_text
    # Basic sanity check
    text = example_target_text("ACC_0001", [], type("A", (), {"total_debit": 0, "total_credit":0, "num_debit":0, "num_credit":0, "unique_creditors":0, "unique_debtors":0, "tx_types":[]}))
    assert "SUSPICIOUS ACTIVITY REPORT" in text
