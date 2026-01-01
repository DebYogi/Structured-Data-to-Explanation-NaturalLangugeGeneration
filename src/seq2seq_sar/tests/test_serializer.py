from seq2seq_sar.data_generator import generate_transactions_for_account
from seq2seq_sar.aggregation import compute_aggregates
from seq2seq_sar.serializer import serialize_account, example_target_text


def test_serialize_and_target():
    txs = generate_transactions_for_account('ACC_X', 2025, 12, 3, seed=1)
    agg = compute_aggregates(txs)
    src = serialize_account('ACC_X', 2025, 12, txs, agg)
    tgt = example_target_text('ACC_X', txs, agg)
    assert 'ACCOUNT_ID=ACC_X' in src
    assert 'PART 1' in tgt
    assert 'PART 2' in tgt
