"""
eval/splits.py — Track E deliverable, laboratory-wide.

Single implementation of every split policy used in Project Regenesis
(Operating Manual §0.1, §3.2: "Splitters, surrogates, metrics, statistical
utilities — Track E — single implementation, laboratory-wide"). No other
track writes into this file or keeps a local copy; a local copy is a
review-blocking defect (§0.1, §11 "no local eval/ utilities").

Every splitter here is BLOCK-WISE. There is deliberately no window-level
option exposed anywhere in this module — window-level splitting is the
laboratory's single most consequential non-negotiable (§0.1: "Inflated
accuracy by tens of points; result withdrawn").

Call these ONE SUBJECT AT A TIME. Pooling subjects into a single split
call is subject-pooled cross-validation reported as cross-subject, which
§2.4 lists explicitly as a leakage form.

Two split policies exist, matching the manual's stated rule ("session-wise
where the dataset provides sessions, else trial-block", §4/§5.1):

  - session_split      — BCI-IV-2a: natural recording sessions exist,
                          so each session is held out in turn.
  - trial_block_split  — WAY-EEG-GAL and anything without a clean session
                          boundary: contiguous acquisition-order blocks
                          are held out instead. Blocks MUST be contiguous
                          in acquisition order — shuffling trials into
                          blocks first defeats the purpose and degrades
                          toward window-level leakage.

get_split() is the dispatcher most pipelines should call; it picks the
right policy automatically instead of leaving the choice to each track.
"""

from __future__ import annotations

from typing import Iterator, Optional

import numpy as np


def session_split(session_ids: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Leave-one-session-out split for a single subject.

    Parameters
    ----------
    session_ids : array-like, shape (n_trials,)
        Session label (int or str) per trial, for ONE subject only.

    Yields
    ------
    (train_idx, test_idx) : tuple of np.ndarray
        Integer indices into the subject's trial axis. One fold per
        unique session, with that session held out in full.
    """
    session_ids = np.asarray(session_ids)
    unique_sessions = np.unique(session_ids)
    if len(unique_sessions) < 2:
        raise ValueError(
            f"session_split requires >=2 sessions, got {len(unique_sessions)}. "
            "This subject/dataset does not provide session structure — "
            "use trial_block_split instead."
        )
    all_idx = np.arange(len(session_ids))
    for held_out in unique_sessions:
        test_mask = session_ids == held_out
        yield all_idx[~test_mask], all_idx[test_mask]


def trial_block_split(
    n_trials: int,
    n_blocks: int = 5,
    block_ids: Optional[np.ndarray] = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Leave-one-block-out split over CONTIGUOUS trial blocks, for a single
    subject.

    Parameters
    ----------
    n_trials : int
        Number of trials for this subject.
    n_blocks : int, default 5
        Number of contiguous blocks to carve [0, n_trials) into, used
        only when `block_ids` is not supplied.
    block_ids : array-like, shape (n_trials,), optional
        Explicit block label per trial, for when block boundaries come
        from the acquisition protocol itself rather than an even split.
        Must still be acquisition-order-contiguous within each block.
        When given, `n_blocks` is ignored.

    Yields
    ------
    (train_idx, test_idx) : tuple of np.ndarray
    """
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


def get_split(
    session_ids: Optional[np.ndarray] = None,
    n_trials: Optional[int] = None,
    n_blocks: int = 5,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Dispatcher: session-wise if session_ids carries >=2 unique sessions,
    else trial-block. This is the call most pipeline code should use —
    it enforces the manual's split policy rule instead of leaving the
    choice to each caller.

    Pass session_ids when your dataset has session labels (BCI-IV-2a);
    pass n_trials (and optionally a smaller n_blocks) when it doesn't
    (WAY-EEG-GAL).
    """
    if session_ids is not None and len(np.unique(np.asarray(session_ids))) >= 2:
        yield from session_split(session_ids)
        return
    if n_trials is None:
        if session_ids is None:
            raise ValueError("get_split needs either session_ids or n_trials.")
        n_trials = len(session_ids)
    yield from trial_block_split(n_trials, n_blocks=n_blocks)