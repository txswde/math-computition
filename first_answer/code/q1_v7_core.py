"""Fold-local candidate response subspace smoothing. Historical files are read-only."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt

from utils import PROJECT_ROOT, sha256_file

CONFIG_PATH = Path(__file__).with_name("q1_v7_config.json")


def config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def folder():
    path = PROJECT_ROOT / config()["output_dir"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_json(obj):
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(name, obj):
    path = folder() / name
    path.write_text(json.dumps(clean_json(obj), ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def save_table(name, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(folder() / name, index=False, encoding="utf-8-sig", float_format="%.17g")
    return frame


def load_data():
    source = PROJECT_ROOT / "results/q12_benchmark"
    with np.load(source / "trial_epochs.npz", allow_pickle=False) as z:
        data = {key: z[key].copy() for key in z.files}
    meta = pd.read_csv(source / "trial_metadata.csv")
    assert len(meta) == 400 and meta.trial_key.is_unique
    assert data["filtered"].shape == (400, 3, 256)
    assert np.isfinite(data["filtered"]).all()
    assert np.array_equal(meta.row_id.to_numpy(), np.arange(400))
    return data, meta


def baseline(x, time):
    return x - x[..., time < 0].mean(axis=-1, keepdims=True)


def filter_extended(extended, extended_time, time, slow=False):
    fs = config()["fs"]
    sos = butter(4, 30., fs=fs, btype="lowpass", output="sos") if slow else \
        butter(4, [.5, 30.], fs=fs, btype="bandpass", output="sos")
    full = sosfiltfilt(sos, extended, axis=-1)
    start = int(round((time[0] - extended_time[0]) * fs))
    return baseline(full[..., start:start + len(time)], time)


def quality_features(native):
    centered = native - native[..., :51].mean(axis=-1, keepdims=True)
    diff = np.diff(centered, axis=-1)
    med = np.median(diff, axis=-1, keepdims=True)
    return np.column_stack([
        np.ptp(centered, axis=-1).max(axis=1),
        np.sqrt(np.mean(centered ** 2, axis=-1)).max(axis=1),
        np.median(np.abs(diff - med), axis=-1).max(axis=1),
        (np.abs(native) >= 999.).mean(axis=-1).max(axis=1),
    ])


@dataclass
class Quality:
    center: np.ndarray
    spread: np.ndarray

    @classmethod
    def fit(cls, q):
        values = np.log1p(np.maximum(q[:, :3], 0))
        center = np.median(values, axis=0)
        spread = np.maximum(1.4826 * np.median(np.abs(values - center), axis=0), 1e-6)
        return cls(center, spread)

    def weights(self, q):
        values = np.log1p(np.maximum(q[:, :3], 0))
        excess = np.maximum((values - self.center) / self.spread - 2., 0)
        w = np.prod(1. / (1. + excess ** 2), axis=1)
        w *= np.clip(1. - 10. * q[:, 3], 0., 1.)
        return np.clip(w, .01, 1.)


@dataclass(frozen=True)
class Candidate:
    name: str
    rank: int
    strength: float
    weighted: bool


def candidates():
    result = [Candidate("P0", 0, 0., False), Candidate("W0", 0, 0., True)]
    for rank in config()["ranks"]:
        for strength in config()["smoothing_lambdas"]:
            result.append(Candidate(f"U{rank}_L{strength:g}", rank, strength, True))
    return result


@lru_cache(maxsize=12)
def smoother(n_time, strength):
    if strength == 0:
        return np.eye(n_time)
    difference = np.diff(np.eye(n_time), n=2, axis=0)
    eigenvalues, eigenvectors = eigh(difference.T @ difference)
    return (eigenvectors / (1. + strength * np.maximum(eigenvalues, 0.))) @ eigenvectors.T


def smooth(x, time, strength):
    if strength == 0:
        return x.copy()
    return baseline(x @ smoother(len(time), float(strength)).T, time)


def mean_by_condition(x, labels, weights):
    values = []
    for sign in [-1, 1]:
        mask = labels == sign
        if not mask.any():
            raise ValueError("Condition missing in training or scoring subset")
        values.append(np.average(x[mask], axis=0, weights=weights[mask]))
    return np.asarray(values)


def fit_basis(x, labels, meta, weights, maximum_rank=4):
    """Training input only. The two columns per block are common and half-contrast."""
    columns = []
    keys = meta.record_id.astype(str) + ":" + ((meta.trial_id - 1) // config()["basis_block_trials"]).astype(str)
    for key in keys.unique():
        idx = np.flatnonzero(keys.to_numpy() == key)
        if np.unique(labels[idx]).size != 2:
            continue
        means = mean_by_condition(x[idx], labels[idx], weights[idx])
        columns.extend([((means[0] + means[1]) / 2).ravel(),
                        ((means[0] - means[1]) / 2).ravel()])
    if not columns:
        return np.zeros((x.shape[1] * x.shape[2], 0)), np.empty(0)
    matrix = np.asarray(columns).T
    # Small Gram eigendecomposition avoids an unnecessary full 768 x 768 SVD.
    values, vectors = eigh(matrix.T @ matrix)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    keep = values > max(float(values[0]), 1.) * 1e-12
    values, vectors = values[keep][:maximum_rank], vectors[:, keep][:, :maximum_rank]
    if len(values) == 0:
        return np.zeros((matrix.shape[0], 0)), values
    basis = matrix @ vectors / np.sqrt(values)[None, :]
    basis, _ = np.linalg.qr(basis, mode="reduced")
    return basis, np.sqrt(values)


@dataclass
class Operator:
    candidate: Candidate
    basis: np.ndarray
    time: np.ndarray
    quality: Quality
    training_keys: list

    def transform(self, x):
        x = np.asarray(x, dtype=float)
        single = x.ndim == 2
        x = x[None] if single else x
        if x.shape[1:] != (3, len(self.time)):
            raise ValueError("Expected trials x 3 channels x time")
        smoothed = smooth(x, self.time, self.candidate.strength)
        u = self.basis[:, :self.candidate.rank]
        if u.shape[1]:
            original = x.reshape(len(x), -1)
            smoothed_flat = smoothed.reshape(len(x), -1)
            out = smoothed_flat + ((original - smoothed_flat) @ u) @ u.T
            out = out.reshape(x.shape)
        else:
            out = smoothed
        return out[0] if single else out

    def weights(self, q):
        return self.quality.weights(q) if self.candidate.weighted else np.ones(len(q))

    def audit(self):
        return {"candidate": asdict(self.candidate), "actual_basis_rank": self.basis.shape[1],
                "training_keys": self.training_keys, "quality_center": self.quality.center,
                "quality_spread": self.quality.spread}


def fit_operator(x, q, labels, meta, time, candidate):
    quality = Quality.fit(q)
    weights = quality.weights(q)
    basis, _ = fit_basis(x, labels, meta, weights, 4)
    return Operator(candidate, basis, time.copy(), quality, meta.trial_key.tolist())


def select_candidate(x, q, labels, meta, time):
    choices = candidates()
    folds = sorted(meta.cv_fold.dropna().unique())
    losses = {c.name: [] for c in choices}
    details = []
    post = (time >= 0) & (time < .8)
    for fold in folds:
        train = np.flatnonzero(meta.cv_fold.to_numpy() != fold)
        valid = np.flatnonzero(meta.cv_fold.to_numpy() == fold)
        if len(train) < 12 or len(valid) < 4 or np.unique(labels[train]).size != 2 or np.unique(labels[valid]).size != 2:
            continue
        prototype = fit_operator(x[train], q[train], labels[train], meta.iloc[train], time, choices[0])
        scale = np.maximum(np.std(x[train], axis=(0, 2)), 1e-6)
        truth = mean_by_condition(x[valid], labels[valid], np.ones(len(valid)))
        for candidate in choices:
            op = Operator(candidate, prototype.basis, time, prototype.quality, prototype.training_keys)
            means = mean_by_condition(x[train], labels[train], op.weights(q[train]))
            prediction = op.transform(means)
            loss = float(np.mean(((prediction[:, :, post] - truth[:, :, post]) / scale[None, :, None]) ** 2))
            losses[candidate.name].append(loss)
            details.append({"inner_fold": int(fold), "candidate": candidate.name, "loss": loss,
                            "n_train": len(train), "n_validation": len(valid)})
    if not any(losses.values()):
        return choices[0], details
    # Deterministic tie handling favors the earlier, simpler candidate.
    chosen = min(choices, key=lambda c: np.mean(losses[c.name]) if losses[c.name] else np.inf)
    return chosen, details


def correlation(a, b):
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def effective_n(weights):
    return float(np.sum(weights) ** 2 / np.sum(weights ** 2))


def protected_paths():
    paths = list((PROJECT_ROOT / "中文题目/C题").glob("*.mat"))
    paths += [p for p in (PROJECT_ROOT / "code").iterdir()
              if p.is_file() and ("v6" in p.name or p.name in ["utils.py", "config.json", "problem1_robust.py"])]
    paths += [p for p in (PROJECT_ROOT / "results/q12_benchmark").iterdir() if p.is_file()]
    paths += [PROJECT_ROOT / "results/trial_table.csv"]
    return {str(p.relative_to(PROJECT_ROOT)): sha256_file(p) for p in sorted(paths)}
