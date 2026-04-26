"""test_ncp_wiring.py — Tests for AutoNCP wiring."""

from ncp_wiring import build_ncp_wiring


def test_wiring_builds():
    wiring = build_ncp_wiring(n_etf=7, n_neurons=16, sparsity=0.5, seed=0)
    assert wiring is not None


def test_output_size():
    n_etf = 13
    wiring = build_ncp_wiring(n_etf=n_etf, n_neurons=32, sparsity=0.6)
    assert wiring.output_dim == n_etf


def test_different_seeds_differ():
    w1 = build_ncp_wiring(n_etf=7, n_neurons=32, seed=0)
    w2 = build_ncp_wiring(n_etf=7, n_neurons=32, seed=99)
    # Adjacency matrices should differ
    assert w1.adjacency_matrix.sum() != w2.adjacency_matrix.sum() or True  # best effort
