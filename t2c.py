"""T2-C: the crosstalk floor.

How much apparent low-dimensionality does the electrode array manufacture
on its own?

Plant N independent muscle sources. Mix them through a model of electrode
crosstalk. Run the normal NMF pipeline. Ask whether it still recovers N.

Everything here is synthetic. That is the point: on real EMG you do not
know the true rank, so the question is unanswerable. Here you planted it.

Run:  python t2c.py --identity-check
      python t2c.py --sweep

Manual references: §5.3 (the experiment), §9.5 (stop-work), §11.1 (this is
the experiment most likely to be quietly deferred).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.decomposition import NMF

# ===========================================================================
# CONFIG - in the real repo this is a YAML committed BEFORE the run (§8.6)
# ===========================================================================

CONFIG = {
    "pr_id": "PR-2026-01",
    "rank_criterion_id": "vaf-threshold-0.95",
    "if_unreached": "undecidable",
    "n_init": 10,
    "max_iter": 2000,
    "planted_ranks": [2, 3, 4, 5],
    "crosstalk": [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    "channel_counts": [8, 12, 16],
    "seeds_per_cell": 20,
    "snr_db": 20.0,
    "fs": 1000.0,
    "n_samples": 4000,
    "band": [20.0, 450.0],
    "env_cutoff": 4.0,
    "filter_order": 4,
    "master_seed": 20260811,
}

RANK_CRITERIA = {
    "vaf-threshold-0.95": 0.95,
    "vaf-threshold-0.90": 0.90,
    "vaf-threshold-0.85": 0.85,
    "vaf-threshold-0.80": 0.80,
}


class RankUndecidable(RuntimeError):
    """No rank in the swept range reached the pre-registered threshold."""


class StopWork(SystemExit):
    """Manual §9.5. Halt and escalate the same day."""


# ===========================================================================
# 1. THE RESULT CONTAINER
# ===========================================================================

@dataclass(frozen=True)
class NMFSolution:
    """W: (channels, rank). H: (rank, time). Columns referred to BY INDEX -
    W[:, 2], never "the power-grip synergy" (§8.3, §11.1)."""

    W: np.ndarray
    H: np.ndarray
    rank: int
    init_seeds: list[int] = field(repr=False)
    vaf_per_init: list[float] = field(repr=False)

    @property
    def vaf(self) -> float:
        return max(self.vaf_per_init)

    @property
    def init_spread(self) -> float:
        """Best minus worst across initialisations. A large spread is a
        finding about the data, not a nuisance to average away (§5.1)."""
        return float(np.ptp(self.vaf_per_init))


# ===========================================================================
# 2. THE MATHS - shared with T2-R and E1. Split this out before the PR.
# ===========================================================================

def vaf(E: np.ndarray, R: np.ndarray) -> float:
    """Variance accounted for, UNCENTERED:  1 - ||E-R||^2 / ||E||^2

    Denominator is sum-of-squares about ZERO, not about the mean. This is
    the synergy literature's convention; sklearn's r2_score uses the
    centered one, and on non-negative envelopes the two differ a lot with
    no error raised either way.
    """
    denom = float(np.sum(E**2))
    if denom == 0.0:
        raise ValueError("cannot compute VAF against an all-zero E")
    return float(1.0 - np.sum((E - R) ** 2) / denom)


def fit_nmf(E: np.ndarray, rank: int, seeds: list[int], max_iter: int) -> NMFSolution:
    """Fit E ~= W @ H at a fixed rank, across many initialisations.

    ORIENTATION: sklearn factorises X (n_samples, n_features). We pass E
    unchanged, so its "samples" are our channels, giving W (channels, rank)
    and H (rank, time) - the synergy convention. Passing E.T runs fine and
    factorises something else entirely.

    The objective is not jointly convex in (W, H), so where you start
    changes where you land. Hence >= 10 inits, every seed recorded.
    """
    if E.min() < 0:
        raise ValueError("NMF requires non-negative input")
    if len(seeds) < 10:
        raise ValueError(f"need >= 10 initialisations, got {len(seeds)}")
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate seeds would understate the spread")

    best, best_v, vafs = None, -np.inf, []
    for s in seeds:
        m = NMF(n_components=rank, init="random", random_state=int(s),
                max_iter=max_iter, tol=1e-5)
        W = m.fit_transform(E)
        H = m.components_
        v = vaf(E, W @ H)
        vafs.append(v)
        if v > best_v:
            best, best_v = (W, H), v

    return NMFSolution(W=best[0], H=best[1], rank=rank,
                       init_seeds=[int(s) for s in seeds], vaf_per_init=vafs)


def select_rank(curve: dict[int, float], criterion_id: str, if_unreached: str) -> int:
    """Apply the PRE-REGISTERED criterion to a {rank: VAF} curve.

    FORBIDDEN: choosing the rank by finding the elbow in the curve. It is
    visible and it feels principled, which is exactly why §11.4 names it.
    """
    if criterion_id not in RANK_CRITERIA:
        raise KeyError(f"unknown criterion {criterion_id!r}; "
                       f"registered: {sorted(RANK_CRITERIA)}")
    threshold = RANK_CRITERIA[criterion_id]

    for rank in sorted(curve):            # sorted, not insertion order
        if curve[rank] >= threshold:      # >= : 0.90 clears a 0.90 bar
            return rank

    if if_unreached == "max_rank":
        return max(curve)
    raise RankUndecidable(
        f"no rank in {sorted(curve)} reached VAF {threshold}; best was "
        f"{max(curve.values()):.3f}. Report undecidable. Do NOT lower the "
        f"threshold - that is choosing after seeing (§11.4)."
    )


def canonicalise(sol: NMFSolution) -> tuple[np.ndarray, np.ndarray]:
    """Unit-norm each W column, push the scale into H. W @ H is unchanged.

    W and H are identified only up to permutation and positive scaling, so
    two independently fitted bases are not comparable until canonicalised.
    """
    n = np.linalg.norm(sol.W, axis=0, keepdims=True)
    safe = np.where(n > 0, n, 1.0)
    return sol.W / safe, sol.H * safe.T


def envelope(x: np.ndarray, fs: float, band, cutoff: float, order: int) -> np.ndarray:
    """Raw signal -> non-negative linear envelope.

    band-pass -> full-wave rectify -> low-pass. Zero-phase (sosfiltfilt)
    so envelope peaks stay aligned with the events that caused them.
    """
    sos_bp = butter(order, band, btype="bandpass", fs=fs, output="sos")
    sos_lp = butter(order, cutoff, btype="lowpass", fs=fs, output="sos")
    y = np.abs(sosfiltfilt(sos_bp, x, axis=-1))
    # filtfilt overshoot can push a rectified signal slightly negative and
    # NMF will refuse it. Clip - do not shift, which changes the VAF
    # denominator.
    return np.clip(sosfiltfilt(sos_lp, y, axis=-1), 0.0, None)


# ===========================================================================
# 3. THE SIMULATOR - this is what makes T2-C T2-C
# ===========================================================================

def planted_activations(rank: int, n_samples: int, fs: float, rng) -> np.ndarray:
    """`rank` activation profiles, INDEPENDENT BY CONSTRUCTION.

    Each source bursts at its own random onsets. No shared drive, no
    coupling. If NMF later reports fewer than `rank` components, that
    reduction was manufactured downstream - which is the whole question.
    """
    t = np.arange(n_samples) / fs
    H = np.zeros((rank, n_samples))
    width = 0.08
    for k in range(rank):
        for onset in rng.uniform(0.2, t[-1] - 0.2, size=6):
            H[k] += np.exp(-((t - onset) ** 2) / (2 * width**2))
    return H


def mixing_matrix(n_channels: int, rank: int, crosstalk: float, rng) -> np.ndarray:
    """Map `rank` sources onto `n_channels` electrodes with neighbour bleed.

    Physical picture: muscle sits in conductive tissue, so an electrode
    over one muscle also hears its neighbours, attenuated with distance.

    crosstalk = 0.0 -> each electrode hears exactly one source (clean)
    crosstalk = 1.0 -> heavy bleed across the array

    The electrodes are laid out on a RING (a forearm), so distance wraps.
    """
    home = np.arange(n_channels) % rank          # which source each channel sits over
    A = np.zeros((n_channels, rank))
    A[np.arange(n_channels), home] = 1.0

    if crosstalk > 0.0:
        pos = np.arange(n_channels)
        d = np.abs(pos[:, None] - pos[None, :])
        d = np.minimum(d, n_channels - d)        # ring distance
        sigma = 0.5 + 3.0 * crosstalk
        bleed = np.exp(-(d**2) / (2 * sigma**2))
        np.fill_diagonal(bleed, 0.0)
        A = A + crosstalk * (bleed @ A)

    gains = 0.7 + 0.6 * rng.random((n_channels, 1))   # electrode-to-electrode gain
    A *= gains
    return A / np.clip(A.sum(axis=1, keepdims=True), 1e-12, None)


def simulate(rank, n_channels, crosstalk, snr_db, cfg, rng) -> np.ndarray:
    """Full forward model: sources -> mixing -> raw EMG -> envelope."""
    H = planted_activations(rank, cfg["n_samples"], cfg["fs"], rng)
    A = mixing_matrix(n_channels, rank, crosstalk, rng)
    drive = A @ H                                       # (channels, time)

    carrier = rng.normal(size=drive.shape)              # broadband muscle activity
    raw = drive * carrier

    sig_p = float(np.mean(raw**2))
    noise_p = sig_p / (10 ** (snr_db / 10.0))
    raw = raw + rng.normal(scale=np.sqrt(noise_p), size=raw.shape)

    return envelope(raw, cfg["fs"], cfg["band"], cfg["env_cutoff"],
                    cfg["filter_order"])


# ===========================================================================
# 4. ONE CELL OF THE EXPERIMENT
# ===========================================================================

def run_cell(planted, n_channels, crosstalk, seed, cfg) -> dict:
    """Simulate once, sweep ranks, apply the criterion. Returns one row."""
    rng = np.random.default_rng(seed)
    E = simulate(planted, n_channels, crosstalk, cfg["snr_db"], cfg, rng)

    max_rank = min(planted + 3, n_channels)
    nmf_seeds = list(range(seed * 100, seed * 100 + cfg["n_init"]))

    curve, spreads = {}, {}
    for r in range(1, max_rank + 1):
        sol = fit_nmf(E, r, nmf_seeds, cfg["max_iter"])
        curve[r] = sol.vaf
        spreads[r] = sol.init_spread

    try:
        recovered = select_rank(curve, cfg["rank_criterion_id"], cfg["if_unreached"])
        undecidable = False
    except RankUndecidable:
        recovered, undecidable = None, True

    return {
        "planted_rank": planted,
        "n_channels": n_channels,
        "crosstalk": crosstalk,
        "seed": seed,
        "recovered_rank": recovered,
        "undecidable": undecidable,
        "vaf_curve": {int(k): round(v, 4) for k, v in curve.items()},
        "init_spread_at_planted": round(spreads.get(planted, float("nan")), 5),
    }


# ===========================================================================
# 5. THE NEGATIVE CONTROL - named in the prereg before any data existed
# ===========================================================================

def identity_check(cfg) -> bool:
    """At zero crosstalk, recovered rank MUST equal planted rank.

    §5.3: if this fails, the pipeline has a bug and EVERY dimensionality
    result in Track B is in question. Stop-work trigger (§9.5).
    """
    print(f"Zero-crosstalk identity check  (SNR {cfg['snr_db']} dB, "
          f"criterion {cfg['rank_criterion_id']})\n")
    print(f"{'planted':>8} {'chans':>6} {'recovered':>10} {'spread':>9}  result")
    print("-" * 52)

    ok = True
    for planted, n_ch in product(cfg["planted_ranks"], cfg["channel_counts"]):
        row = run_cell(planted, n_ch, 0.0, cfg["master_seed"], cfg)
        passed = row["recovered_rank"] == planted
        ok &= passed
        print(f"{planted:>8} {n_ch:>6} {str(row['recovered_rank']):>10} "
              f"{row['init_spread_at_planted']:>9.5f}  {'PASS' if passed else 'FAIL'}")

    print()
    if not ok:
        raise StopWork(
            "\nSTOP-WORK (§9.5): zero-crosstalk identity check FAILED.\n"
            "At zero crosstalk there is nothing to confuse the pipeline, so a\n"
            "mismatch means the pipeline itself is wrong - not the simulator.\n"
            "Every dimensionality result in Track B is in question until this\n"
            "is resolved. Escalate today.\n\n"
            "Check, in order: the VAF definition (uncentered?), the envelope\n"
            "parameters, the swept rank range, the initialisation count.\n"
            "Do NOT resolve this by adjusting the threshold or the SNR - both\n"
            "are pre-registered, and tuning them now is tuning after seeing.\n"
        )
    print("Identity check PASSED. The pipeline recovers planted rank on clean data.")
    return ok


# ===========================================================================
# 6. THE FULL SWEEP
# ===========================================================================

def run_sweep(cfg, out_path: Path) -> list[dict]:
    cells = list(product(cfg["planted_ranks"], cfg["channel_counts"], cfg["crosstalk"]))
    total = len(cells) * cfg["seeds_per_cell"]
    print(f"{total} runs across {len(cells)} cells\n")

    rows, done = [], 0
    for planted, n_ch, ct in cells:
        for i in range(cfg["seeds_per_cell"]):
            rows.append(run_cell(planted, n_ch, ct, cfg["master_seed"] + i, cfg))
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{total}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"config": cfg, "rows": rows}, indent=2))
    print(f"\nwrote {out_path}")
    return rows


def summarise(rows: list[dict]) -> None:
    """Recovered rank as a DISTRIBUTION across seeds, not a single value (§5.3)."""
    print(f"\n{'planted':>8} {'chans':>6} {'crosstalk':>10} "
          f"{'median':>7} {'min':>5} {'max':>5} {'undec':>6}")
    print("-" * 56)
    key = lambda r: (r["planted_rank"], r["n_channels"], r["crosstalk"])
    for k in sorted({key(r) for r in rows}):
        group = [r for r in rows if key(r) == k]
        rec = [r["recovered_rank"] for r in group if r["recovered_rank"] is not None]
        undec = sum(r["undecidable"] for r in group)
        med = f"{np.median(rec):.1f}" if rec else "-"
        lo = min(rec) if rec else "-"
        hi = max(rec) if rec else "-"
        print(f"{k[0]:>8} {k[1]:>6} {k[2]:>10.2f} {med:>7} {lo:>5} {hi:>5} {undec:>6}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="T2-C: the crosstalk floor")
    p.add_argument("--identity-check", action="store_true",
                   help="run the zero-crosstalk negative control only")
    p.add_argument("--sweep", action="store_true", help="run the full sweep")
    p.add_argument("--out", type=Path, default=Path("results/PR-2026-01/raw/t2c.json"))
    args = p.parse_args()

    if args.identity_check:
        identity_check(CONFIG)
    elif args.sweep:
        identity_check(CONFIG)          # the control gates the sweep
        summarise(run_sweep(CONFIG, args.out))
    else:
        p.print_help()
