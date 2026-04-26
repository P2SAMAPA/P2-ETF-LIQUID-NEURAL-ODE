"""config.py — Pydantic BaseSettings for P2-ETF-LIQUID-NEURAL-ODE.

Loads ltc_config.toml (or a custom path) and validates all hyperparameters.
"""

from __future__ import annotations

import sys
from pathlib import Path

# tomllib is stdlib in 3.11+; fall back to tomli on 3.10
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

from pydantic import BaseModel, Field, field_validator

# ── Sub-models ────────────────────────────────────────────────────────────────


class ModelConfig(BaseModel):
    n_neurons: int = 64
    n_sensory: int = 32
    n_inter: int = 24
    n_command: int = 16
    tau_min: float = 0.1
    tau_max: float = 10.0
    sparsity: float = 0.6
    use_closed_form: bool = False
    input_dim: int = 14
    dropout: float = 0.1

    @field_validator("sparsity")
    @classmethod
    def check_sparsity(cls, v: float) -> float:
        assert 0.0 < v < 1.0, "sparsity must be in (0, 1)"
        return v


class TrainingConfig(BaseModel):
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 300
    patience: int = 25
    batch_size: int = 32
    grad_clip: float = 1.0
    sharpe_lambda: float = 0.1
    warmup_epochs: int = 5
    seed: int = 42


class ODEConfig(BaseModel):
    method: str = "dopri5"
    rtol: float = 1e-3
    atol: float = 1e-4
    adjoint: bool = True


class DataConfig(BaseModel):
    window: int = 63
    hf_repo: str = "P2SAMAPA/fi-etf-macro-signal-master-data"
    parquet_file: str = "master_data.parquet"
    train_end: str = "2019-12-31"
    val_end: str = "2022-12-31"


class ScoringConfig(BaseModel):
    alpha_amp: float = 0.3
    mc_passes: int = 50
    ci_level: float = 0.95


class OutputConfig(BaseModel):
    hf_results_repo: str = "P2SAMAPA/p2-etf-liquid-neural-ode-results"
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"


# ── Root config ───────────────────────────────────────────────────────────────


class LTCConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    ode: ODEConfig = Field(default_factory=ODEConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


def load_config(path: str | Path = "ltc_config.toml") -> LTCConfig:
    """Load and validate config from a TOML file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, "rb") as f:
        raw = tomllib.load(f)
    return LTCConfig(**raw)
