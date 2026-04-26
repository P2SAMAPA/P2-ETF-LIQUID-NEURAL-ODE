"""ncp_wiring.py — Neural Circuit Policy (AutoNCP) wiring for LTC model.

Sensory(32) → Inter(24) → Command(16) → Motor(N_etf)
Sparsity ≈ 0.6  (40% of possible connections active)
"""

from __future__ import annotations

from ncps.wirings import AutoNCP

from logging_utils import get_logger

log = get_logger(__name__)


def build_ncp_wiring(
    n_etf: int,
    n_neurons: int = 64,
    n_sensory: int = 32,
    n_inter: int = 24,
    n_command: int = 16,
    sparsity: float = 0.6,
    seed: int = 42,
) -> AutoNCP:
    """Build an AutoNCP wiring object.

    AutoNCP automatically generates biologically-inspired sparse connectivity
    between Sensory → Inter → Command → Motor neuron layers.

    Args:
        n_etf:      Number of motor neurons (one per ETF in universe).
        n_neurons:  Total inter+command neuron count.
        n_sensory:  Sensory neuron count (receive input x).
        sparsity:   Fraction of possible connections that are *inactive*.
        seed:       Random seed for wiring reproducibility.

    Returns:
        AutoNCP wiring instance ready to pass to ncps.torch.LTC.
    """
    # AutoNCP: units = total neurons including motor; out_features = motor neurons
    wiring = AutoNCP(
        units=n_neurons,
        output_size=n_etf,
        sparsity_level=sparsity,
        seed=seed,
    )
    log.info(
        "NCP wiring: neurons=%d → motor=%d  sparsity=%.2f",
        n_neurons,
        n_etf,
        sparsity,
    )
    return wiring
