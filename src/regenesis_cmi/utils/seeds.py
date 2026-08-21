"""Deterministic seed derivation without mutable global RNG state."""

from __future__ import annotations

import hashlib


def derive_seed(base_seed: int, *labels: object) -> int:
    payload = "|".join((str(base_seed), *(str(label) for label in labels)))
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)

