"""Common result contract for every candidate estimator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Direction = Literal["I(G;E|M)", "I(G;M|E)", "generic"]


@dataclass(frozen=True)
class EstimatorResult:
    """Auditable output shared by exact, direct, and decoder estimators."""

    estimate_bits: float
    direction: Direction | str
    estimator_name: str
    sample_size: int
    seed: int
    runtime_seconds: float
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    units: str = "bits per independent trial"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

