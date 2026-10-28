"""Benchmark invariants; GitHub Actions also executes actual paired experiments."""
import numpy as np
import pytest
import torch

from benchmarks.dp_utility_gap import Protocol, load_public_digits, client_indices, datasets_for_clients, plain_client_round
from src import fl


def test_public_data_split_is_disjoint_binary_and_reproducible():
    a = load_public_digits(11)
    b = load_public_digits(11)
    assert len(a[0]) + len(a[3]) == 1797
    assert a[0].shape[1] == 64
    assert not set(a[2]) & set(a[5])
    assert set(np.unique(a[1])) == {0., 1.}
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("scenario", ["iid", "label_dirichlet"])
def test_public_partition_is_disjoint_complete_and_repeatable(scenario):
    X, y, ids, *_ = load_public_digits(11)
    a = client_indices(y, 3, scenario, 1020)
    b = client_indices(y, 3, scenario, 1020)
    assert min(map(len, a)) >= 20
    assert all(np.array_equal(x, z) for x, z in zip(a, b))
    assert np.array_equal(np.sort(np.concatenate(a)), np.arange(len(y)))
    ds = datasets_for_clients(X, y, ids, a)
    assert len(ds) == 3 and sum(len(d) for d in ds) == len(y)
    assert len(set(np.concatenate([d.record_ids for d in ds]))) == len(y)


def test_label_dirichlet_is_more_skewed_than_iid():
    _, y, *_ = load_public_digits(11)
    iid = client_indices(y, 3, "iid", 1020)
    skew = client_indices(y, 3, "label_dirichlet", 1020)
    assert np.std([y[g].mean() for g in skew]) > np.std([y[g].mean() for g in iid)


def test_bad_splits_are_rejected():
    X, y, ids, *_ = load_public_digits(11)
    with pytest.raises(ValueError):
        datasets_for_clients(X, y, ids, [np.array([0, 0]), np.array([1])])
    with pytest.raises(ValueError):
        client_indices(y[:20], 3, "iid", 11)
    with pytest.raises(ValueError):
        client_indices(y, 3, "invalid", 11)


def test_plain_training_does_not_mutate_state_and_is_repeatable():
    torch.set_num_threads(1)
    X, y, ids, *_ = load_public_digits(11)
    data = datasets_for_clients(X, y, ids, client_indices(y, 3, "iid", 1020))[0]
    cfg = Protocol(rounds=1, local_steps=2).dp_config()
    state = fl.get_state(fl.make_model(64, seed=11))
    before = {k: v.clone() for k, v in state.items()}
    first = plain_client_round(state, data, cfg, 64, 18)
    again = plain_client_round(state, data, cfg, 64, 18)
    assert all(torch.equal(v, state[k]) for k, v in before.items())
    assert all(torch.equal(first[k], again[k]) for k in first)
    assert any(not torch.equal(first[k], state[k]) for k in first)
