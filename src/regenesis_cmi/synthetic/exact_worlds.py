"""Generator A: discrete worlds whose information is analytically known."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ExactWorld:
    name: str
    g: NDArray[np.int64]
    eeg: NDArray[np.int64]
    emg: NDArray[np.int64]
    truth_forward_bits: float
    truth_reverse_bits: float
    description: str


def binary_entropy(q: float) -> float:
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must lie in [0, 1]")
    if q in (0.0, 1.0):
        return 0.0
    return float(-q * np.log2(q) - (1.0 - q) * np.log2(1.0 - q))


def complementary_exact(repeats: int = 1) -> ExactWorld:
    pairs = np.asarray(list(product((0, 1), repeat=2)), dtype=int)
    pairs = np.tile(pairs, (repeats, 1))
    a, b = pairs[:, 0], pairs[:, 1]
    return ExactWorld(
        name="complementary_four_class",
        g=(2 * a + b).astype(np.int64),
        emg=a.astype(np.int64),
        eeg=b.astype(np.int64),
        truth_forward_bits=1.0,
        truth_reverse_bits=1.0,
        description="M reveals A and E reveals the independent bit B.",
    )


def complementary_sampled(n_samples: int, seed: int = 0) -> ExactWorld:
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 2, n_samples, dtype=np.int64)
    b = rng.integers(0, 2, n_samples, dtype=np.int64)
    return ExactWorld(
        name="complementary_four_class_sampled",
        g=2 * a + b,
        emg=a,
        eeg=b,
        truth_forward_bits=1.0,
        truth_reverse_bits=1.0,
        description="Sampled counterpart of the complementary four-class world.",
    )


def noisy_complementary_exact(q: float, denominator: int = 100) -> ExactWorld:
    flips = int(round(q * denominator))
    realized_q = flips / denominator
    if not np.isclose(realized_q, q, atol=1e-12):
        raise ValueError("q must be representable at the requested denominator")
    rows: list[tuple[int, int, int]] = []
    for a, b in product((0, 1), repeat=2):
        rows.extend((2 * a + b, b, a) for _ in range(denominator - flips))
        rows.extend((2 * a + b, 1 - b, a) for _ in range(flips))
    array = np.asarray(rows, dtype=np.int64)
    return ExactWorld(
        name=f"noisy_complementary_q_{q:.2f}",
        g=array[:, 0],
        eeg=array[:, 1],
        emg=array[:, 2],
        truth_forward_bits=1.0 - binary_entropy(q),
        # E is a noisy B while M reveals A, so M still contributes one full bit.
        truth_reverse_bits=1.0,
        description="E is B passed through a binary symmetric channel.",
    )


def noisy_complementary_sampled(
    n_samples: int, q: float, seed: int = 0
) -> ExactWorld:
    base = complementary_sampled(n_samples, seed)
    rng = np.random.default_rng(seed + 1)
    flips = rng.random(n_samples) < q
    eeg = np.bitwise_xor(base.eeg, flips.astype(np.int64))
    return ExactWorld(
        name=f"noisy_complementary_sampled_q_{q:.2f}",
        g=base.g,
        eeg=eeg,
        emg=base.emg,
        truth_forward_bits=1.0 - binary_entropy(q),
        truth_reverse_bits=1.0,
        description="Finite sampled binary-symmetric complementary channel.",
    )


def redundant_markov_exact() -> ExactWorld:
    # Rational noise probabilities yield a finite exact empirical distribution.
    rows: list[tuple[int, int, int]] = []
    for g in (0, 1):
        for nm in ([0] * 7 + [1]):  # P(M != G) = 1/8
            m = g ^ nm
            for ne in ([0] * 3 + [1]):  # P(E != M) = 1/4
                rows.append((g, m ^ ne, m))
    array = np.asarray(rows, dtype=np.int64)
    return ExactWorld(
        name="redundant_markov_chain",
        g=array[:, 0],
        eeg=array[:, 1],
        emg=array[:, 2],
        truth_forward_bits=0.0,
        truth_reverse_bits=float("nan"),
        description="G -> observed M -> E; E is informative but conditionally redundant.",
    )


def asymmetric_exact() -> ExactWorld:
    triples = np.asarray(list(product((0, 1), repeat=3)), dtype=np.int64)
    a, b, c = triples[:, 0], triples[:, 1], triples[:, 2]
    g = 4 * a + 2 * b + c
    emg = np.column_stack((a, b))
    eeg = c
    return ExactWorld(
        name="asymmetric_three_bit",
        g=g,
        eeg=eeg,
        emg=emg,
        truth_forward_bits=1.0,
        truth_reverse_bits=2.0,
        description="M reveals A,B and E reveals C, proving direction asymmetry.",
    )


def all_exact_worlds() -> list[ExactWorld]:
    return [complementary_exact(25), redundant_markov_exact(), asymmetric_exact()]

