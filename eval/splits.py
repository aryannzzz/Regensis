"""eval/splits.py — Track E deliverable. See repo copy for full docstring."""
from __future__ import annotations
from typing import Iterator, Optional
import numpy as np


def session_split(session_ids: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    session_ids = np.asarray(session_ids)
    unique_sessions = np.unique(session_ids)
    if len(unique_sessions) < 2:
        raise ValueError(
            f"session_split requires >=2 sessions, got {len(unique_sessions)}."
        )
    all_idx = np.arange(len(session_ids))
    for held_out in unique_sessions:
        test_mask = session_ids == held_out
        yield all_idx[~test_mask], all_idx[test_mask]


def trial_block_split(n_trials, n_blocks=5, block_ids=None):
    all_idx = np.arange(n_trials)
    if block_ids is None:
        if n_blocks < 2:
            raise ValueError("n_blocks must be >= 2.")
        chunks = np.array_split(all_idx, n_blocks)
        block_ids = np.concatenate(
            [np.full(len(chunk), i) for i, chunk in enumerate(chunks)]
        )
    else:
        block_ids = np.asarray(block_ids)
        if len(block_ids) != n_trials:
            raise ValueError("block_ids length must equal n_trials.")
    unique_blocks = np.unique(block_ids)
    if len(unique_blocks) < 2:
        raise ValueError("Need >=2 blocks for a leave-one-block-out split.")
    for held_out in unique_blocks:
        test_mask = block_ids == held_out
        yield all_idx[~test_mask], all_idx[test_mask]


def get_split(session_ids=None, n_trials=None, n_blocks=5):
    if session_ids is not None and len(np.unique(np.asarray(session_ids))) >= 2:
        yield from session_split(session_ids)
        return
    if n_trials is None:
        if session_ids is None:
            raise ValueError("get_split needs either session_ids or n_trials.")
        n_trials = len(session_ids)
    yield from trial_block_split(n_trials, n_blocks=n_blocks)