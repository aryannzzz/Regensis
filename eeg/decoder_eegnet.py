"""
eeg/decoder_eegnet.py

Compact EEGNet-style CNN, the "main model" detector arm for E3-A
(protocol doc pipeline step 5: "sLDA (background baseline) and EEGNet
(main model)"). Consumes raw windowed MRCP signal (n_channels,
window_len) directly -- unlike eeg/anticipation.py's sLDA path, which
runs on preprocessing/features_eeg.py's hand-crafted mean/slope
features. That split is deliberate: sLDA is the simple baseline,
EEGNet is the one architecture allowed to learn its own spatio-temporal
filters from raw signal (manual S: "why a small, structured architecture
outperforms larger ones on small EEG datasets -- the design reasoning
matters more than the architecture").

Interface mirrors eeg/decoders_linear.py's shape (a Config dataclass +
build + fit_predict_fold) so experiments/run_e3a.py can select the
detector family by config alone, same pattern as run_t1r.py does for
csp vs. riemann.

Requires torch (not yet in env.lock -- add `torch` before this is run
for a real E3-A result, per S8.7/env.lock's role as part of the
reproducibility floor). Guarded import, same pattern as
data/bci_iv_2a.py's mne/scipy guards.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as e:
    raise ImportError(
        "eeg/decoder_eegnet.py requires torch. Install with: pip install torch "
        "-- and add it to env.lock before using this for any real result."
    ) from e


@dataclass
class EEGNetConfig:
    n_channels: int = 32
    window_len: int = 250            # samples; set from
                                      # SlidingWindowConfig.window_s * fs
    f1: int = 8                      # temporal filters
    d: int = 2                       # depth multiplier (depthwise spatial filters per temporal filter)
    f2: int = 16                     # separable-conv pointwise filters (usually f1*d)
    kernel_length: int = 64          # temporal kernel, samples (~half a cycle
                                      # of the MRCP band's lowest frequency
                                      # at the given fs -- set per fs, not
                                      # left at a fixed default across configs)
    dropout: float = 0.5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 100
    batch_size: int = 32
    patience: int = 10               # early-stopping patience on a held-in
                                      # validation slice of the training fold
                                      # only -- never the evaluation fold
                                      # (S0.1: hyperparameters/early-stopping
                                      # selected on a nested inner fold)
    seed: int = 0


class _EEGNet(nn.Module):
    """Standard EEGNet (Lawhern et al. 2018) block structure: temporal
    conv -> depthwise spatial conv -> separable conv -> classifier."""

    def __init__(self, cfg: EEGNetConfig):
        super().__init__()
        self.cfg = cfg
        c, t = cfg.n_channels, cfg.window_len

        self.block1 = nn.Sequential(
            nn.Conv2d(1, cfg.f1, (1, cfg.kernel_length), padding=(0, cfg.kernel_length // 2), bias=False),
            nn.BatchNorm2d(cfg.f1),
            nn.Conv2d(cfg.f1, cfg.f1 * cfg.d, (c, 1), groups=cfg.f1, bias=False),  # depthwise spatial
            nn.BatchNorm2d(cfg.f1 * cfg.d),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(cfg.dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(cfg.f1 * cfg.d, cfg.f1 * cfg.d, (1, 16), padding=(0, 8),
                      groups=cfg.f1 * cfg.d, bias=False),  # depthwise
            nn.Conv2d(cfg.f1 * cfg.d, cfg.f2, (1, 1), bias=False),  # pointwise -> separable conv
            nn.BatchNorm2d(cfg.f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(cfg.dropout),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, c, t)
            flat_dim = self.block2(self.block1(dummy)).flatten(1).shape[1]
        self.classifier = nn.Linear(flat_dim, 1)

    def forward(self, x):
        # x: (batch, n_channels, window_len) -> (batch, 1, n_channels, window_len)
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(1)
        return self.classifier(x).squeeze(-1)  # logit


def build_model(cfg: EEGNetConfig) -> _EEGNet:
    torch.manual_seed(cfg.seed)
    return _EEGNet(cfg)


def fit_predict_fold(cfg: EEGNetConfig, X_train, y_train, X_test, y_test,
                      X_val=None, y_val=None):
    """
    One fold. X_* shape (n_windows, n_channels, window_len); y_* in {0,1}.

    If X_val/y_val given, they're the nested inner-fold validation slice
    for early stopping (S0.1: hyperparameters/early-stopping selected on
    a nested inner fold, never the evaluation fold). If not given, a
    fraction of X_train is held out internally for this purpose --
    X_test is never touched for early stopping either way.

    Returns (predictions, decision_values) for X_test, matching
    eeg/decoders_linear.fit_predict_fold's return shape so
    eeg/anticipation.py can treat both detector families identically.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    if X_val is None:
        rng = np.random.default_rng(cfg.seed)
        idx = rng.permutation(len(X_train))
        n_val = max(1, int(0.15 * len(X_train)))
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        X_val, y_val = X_train[val_idx], y_train[val_idx]
        X_train, y_train = X_train[train_idx], y_train[train_idx]

    def _tensor(a, dtype=torch.float32):
        return torch.as_tensor(np.asarray(a), dtype=dtype, device=device)

    Xtr, ytr = _tensor(X_train), _tensor(y_train)
    Xval, yval = _tensor(X_val), _tensor(y_val)
    Xte = _tensor(X_test)

    best_val_loss = float("inf")
    best_state = None
    epochs_since_improvement = 0
    n = len(Xtr)

    for epoch in range(cfg.max_epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for start in range(0, n, cfg.batch_size):
            batch_idx = perm[start:start + cfg.batch_size]
            opt.zero_grad()
            logits = model(Xtr[batch_idx])
            loss = loss_fn(logits, ytr[batch_idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xval), yval).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        decision_values = model(Xte).cpu().numpy()
    predictions = (decision_values > 0).astype(int)
    return predictions, decision_values