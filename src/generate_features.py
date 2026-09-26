"""
Dataset v3 generator: v1 features (unchanged, bit-identical) + windowing additions. Self-contained.

    python generate_dataset_v3.py --x-path X_train.parquet --y-path y_train.parquet \
        --y-index-path y_index.parquet --output-path data/feat_v3.parquet --num-workers 6

Sections
  1. FEATURE ENGINE  (numpy + scipy only; paste this section into the submission for infer/train)
  2. DATASET CLI     (pandas / pyarrow / click / tqdm; offline only)

v3 additions over v1 (35 features):
  A. local-baseline detectors : x, x2, xx, v re-calibrated on the LAST 250 history points
                                -> loc_{sig}_scan_max, loc_{sig}_full, loc_{sig}_dec_t          (12)
  B. history-tail context     : tail_var_ratio, tail_mean, tail_acf1_diff                        (3)
  C. short online windows     : {x, x2, xx, v, z2}_win{10,25,50} = window sum / sqrt(w)          (15)
  D. small-shift CUSUM + trend: x2/v CUSUM up/dn with k=0.25 (running max), best_scan_max_d5    (5)
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import math
from collections import deque

import numpy as np
from scipy.signal import lfilter
from scipy.special import ndtri


# =====================================================================================
# 1. FEATURE ENGINE
# =====================================================================================
EPS = 1e-12
AR_ORDER = 3
CUSUM_K = 0.5
MIN_SEG = 5
THRESH = 3.0
T_SCALE = 0.25
M_GRID = np.asarray([5, 10, 25, 50, 100, 250], dtype=np.float64)
SWING_Q = (0.005, 0.995)
LOC_WIN = 20

# v3 settings
TAIL_LEN = 250                                   # "recent history" = last 250 history points
LOCAL_SIGNALS = ("x", "x2", "xx", "v")           # group A
WINDOW_SIGNALS = ("x", "x2", "xx", "v", "z2")    # group C
WINDOWS = (10, 25, 50)
SMALL_CUSUM_K = 0.25                             # group D
SMALL_CUSUM_SIGNALS = ("x2", "v")
TREND_LAG = 5

SIGNAL_NAMES = ("x", "e", "z", "v", "z2", "d", "x2", "xx")
PER_SIGNAL_NAMES = (
    "scan",
    "scan_max",
    "scan_dir",
    "scan_age",
    "cusum_up_max",
    "cusum_dn_max",
    "dec_t",
    "full",
    "swing",
    "since_cross3",
)
_SIG_IDX = {name: i for i, name in enumerate(SIGNAL_NAMES)}


# ----------------------------- helpers -----------------------------
def _safe_std(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return EPS
    value = float(np.std(x, ddof=0))
    return max(value, EPS) if np.isfinite(value) else EPS


def _kurtosis_nonexcess(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size < 2:
        return 3.0
    centered = x - np.mean(x)
    m2 = np.mean(centered * centered)
    if not np.isfinite(m2) or m2 <= EPS:
        return 3.0
    value = np.mean(centered ** 4) / (m2 * m2)
    return float(value) if np.isfinite(value) else 3.0


def _corr1(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size < 3:
        return 0.0
    a, b = x[1:], x[:-1]
    if np.std(a) <= EPS or np.std(b) <= EPS:
        return 0.0
    value = np.corrcoef(a, b)[0, 1]
    return float(value) if np.isfinite(value) else 0.0


def _fit_ar(history, order=AR_ORDER):
    history = np.asarray(history, dtype=np.float64)
    if history.size <= order:
        raise ValueError(f"History needs more than {order} values for AR({order}).")
    y = history[order:]
    X = np.column_stack(
        [np.ones(y.size)] + [history[order - lag: history.size - lag] for lag in range(1, order + 1)]
    )
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    sigma_e = _safe_std(residuals)
    return beta, residuals, sigma_e, residuals / sigma_e


def _rank_table(n):
    """Normal score for each possible right-rank 0..n (same values as v1's per-point ndtri)."""
    u = np.clip((np.arange(n + 1, dtype=np.float64) + 0.5) / (n + 1.0), 1e-6, 1.0 - 1e-6)
    return ndtri(u)


def _newey_west_std(x):
    """Bartlett-kernel long-run std with automatic bandwidth."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return 1.0
    xc = x - np.mean(x)
    gamma0 = float(np.mean(xc * xc))
    if not np.isfinite(gamma0) or gamma0 <= EPS:
        return np.sqrt(EPS)
    bandwidth = min(max(int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))), 0), n - 1)
    lrv = gamma0
    for lag in range(1, bandwidth + 1):
        gamma = float(np.mean(xc[lag:] * xc[:-lag]))
        lrv += 2.0 * (1.0 - lag / (bandwidth + 1.0)) * gamma
    floor = max(0.1 * gamma0, EPS)
    lrv = max(lrv, floor) if np.isfinite(lrv) else floor
    return float(np.sqrt(lrv))


def _fixed_decay_series(c_hist, memory):
    """Normalised EWMA A_t / sqrt(B_t), vectorised (A_t = a A_{t-1} + c_t, B_t = a^2 B_{t-1} + 1)."""
    c_hist = np.asarray(c_hist, dtype=np.float64)
    a = np.exp(-1.0 / float(memory))
    A = lfilter([1.0], [1.0, -a], c_hist)
    B = lfilter([1.0], [1.0, -a * a], np.ones_like(c_hist))
    return A / np.sqrt(np.maximum(B, EPS))


def _build_swing_reference(c_hist):
    refs_m, refs_hi, refs_lo = [], [], []
    for memory in M_GRID:
        values = _fixed_decay_series(c_hist, memory)
        burn = int(3 * memory)
        if values.size <= burn:
            continue
        lo, hi = np.quantile(values[burn:], SWING_Q)
        if np.isfinite(lo) and np.isfinite(hi):
            refs_m.append(float(memory))
            refs_hi.append(float(hi))
            refs_lo.append(float(lo))
    if not refs_m:  # short history: fall back to the smallest memory, no burn-in
        memory = float(M_GRID[0])
        lo, hi = np.quantile(_fixed_decay_series(c_hist, memory), SWING_Q)
        refs_m, refs_hi, refs_lo = [memory], [float(hi)], [float(lo)]
    return (
        np.asarray(refs_m, dtype=np.float64),
        np.asarray(refs_hi, dtype=np.float64),
        np.asarray(refs_lo, dtype=np.float64),
    )


# ----------------------------- history model -----------------------------
def _history_model(history):
    history = np.asarray(history, dtype=np.float64)
    beta, residuals, sigma_e, e_hist = _fit_ar(history)
    h_sorted = np.sort(history)
    z_table = _rank_table(len(h_sorted))

    h_core = history[AR_ORDER:]
    z_hist = z_table[np.searchsorted(h_sorted, h_core, side="right")]
    e_prev_hist = np.r_[0.0, e_hist[:-1]]
    x_prev_hist = history[AR_ORDER - 1: -1]

    raw_hist = {
        "x": h_core,
        "e": e_hist,
        "z": z_hist,
        "v": e_hist ** 2,
        "z2": z_hist ** 2,
        "d": e_hist * e_prev_hist,
        "x2": h_core ** 2,
        "xx": h_core * x_prev_hist,
    }

    # v1 calibration: whole history
    calibrations, swing_refs = {}, {}
    for name in SIGNAL_NAMES:
        values = np.asarray(raw_hist[name], dtype=np.float64)
        mu = float(np.mean(values))
        sigma = _newey_west_std(values)
        c_hist = (values - mu) / max(sigma, EPS)
        c_hist[~np.isfinite(c_hist)] = 0.0
        calibrations[name] = (mu, sigma)
        swing_refs[name] = _build_swing_reference(c_hist)

    # v3 group A: local calibration on the last TAIL_LEN points of each history signal
    local_calibrations = {}
    for name in LOCAL_SIGNALS:
        tail = np.asarray(raw_hist[name][-TAIL_LEN:], dtype=np.float64)
        local_calibrations[name] = (float(np.mean(tail)), _newey_west_std(tail))

    abs_h = np.abs(history)
    q99 = float(np.quantile(abs_h, 0.99))
    hmax = max(float(np.max(abs_h)), EPS)

    context = {
        "hist_length": float(len(history)),
        "ar1": float(beta[1]),
        "ar_sum": float(np.sum(beta[1:])),
        "resid_ratio": float(sigma_e / _safe_std(history)),
        "kurt": _kurtosis_nonexcess(e_hist),
        "arch1": _corr1(e_hist ** 2),
        "hist_q99": q99,
        "hist_max_abs": hmax,
    }

    # v3 group B: where did the history end up?
    h_tail = history[-TAIL_LEN:]
    tail_context = {
        "tail_var_ratio": float(np.var(h_tail) / max(np.var(history), EPS)),
        "tail_mean": float(np.mean(h_tail)),
        "tail_acf1_diff": _corr1(h_tail) - _corr1(history),
    }

    return {
        "beta": beta,
        "sigma_e": sigma_e,
        "h_sorted": h_sorted,
        "z_table": z_table,
        "q99": q99,
        "hmax": hmax,
        "lags": list(history[-AR_ORDER:][::-1]),  # most recent first
        "e_prev": float(e_hist[-1]),
        "x_prev": float(history[-1]),
        "calibrations": calibrations,
        "local_calibrations": local_calibrations,
        "swing_refs": swing_refs,
        "context": context,
        "tail_context": tail_context,
    }


# ----------------------------- detectors for all signals at once -----------------------------
class SignalBank:
    """v1 per-signal detectors, all signals as matrix rows (same arithmetic as v1, faster)."""

    def __init__(self, swing_refs, capacity=256):
        k = len(SIGNAL_NAMES)
        self.k = k
        self.rows = np.arange(k)
        self.S = np.zeros((k, capacity + 1), dtype=np.float64)   # S[:, t] = sum of first t values
        self.C = np.zeros((k, capacity), dtype=np.float64)
        self.cusum_up = np.zeros(k)
        self.cusum_dn = np.zeros(k)
        self.cusum_up_max = np.zeros(k)
        self.cusum_dn_max = np.zeros(k)
        self.scan_max = np.zeros(k)
        self.first_cross3 = np.zeros(k, dtype=np.int64)           # 0 = never crossed
        self.above3_count = np.zeros(k, dtype=np.int64)
        self.swing_log_m = [np.log(swing_refs[n][0]).tolist() for n in SIGNAL_NAMES]
        self.swing_m_lo = [float(swing_refs[n][0][0]) for n in SIGNAL_NAMES]
        self.swing_m_hi = [float(swing_refs[n][0][-1]) for n in SIGNAL_NAMES]
        self.swing_hi = [swing_refs[n][1].tolist() for n in SIGNAL_NAMES]
        self.swing_lo = [swing_refs[n][2].tolist() for n in SIGNAL_NAMES]

    def _grow(self, t):
        if t >= self.C.shape[1]:
            self.C = np.concatenate([self.C, np.zeros_like(self.C)], axis=1)
            self.S = np.concatenate([self.S, np.zeros((self.k, self.S.shape[1]))], axis=1)

    def _swing_ref(self, i, memory):
        log_m = self.swing_log_m[i]
        hi, lo = self.swing_hi[i], self.swing_lo[i]
        if len(log_m) == 1:
            return hi[0], lo[0]
        q = math.log(min(max(memory, self.swing_m_lo[i]), self.swing_m_hi[i]))
        j = 0
        while j < len(log_m) - 2 and q > log_m[j + 1]:
            j += 1
        w = (q - log_m[j]) / (log_m[j + 1] - log_m[j])
        return hi[j] + w * (hi[j + 1] - hi[j]), lo[j] + w * (lo[j + 1] - lo[j])

    def update(self, c, t, step):
        self._grow(t)
        self.C[:, t - 1] = c
        S_t = self.S[:, t - 1] + c
        self.S[:, t] = S_t

        self.cusum_up = np.maximum(0.0, self.cusum_up + c - CUSUM_K)
        self.cusum_dn = np.maximum(0.0, self.cusum_dn - c - CUSUM_K)
        self.cusum_up_max = np.maximum(self.cusum_up_max, self.cusum_up)
        self.cusum_dn_max = np.maximum(self.cusum_dn_max, self.cusum_dn)

        if t < MIN_SEG:
            scan = np.abs(S_t) / math.sqrt(t)
            j_star = np.zeros(self.k, dtype=np.int64)
            delta = S_t
        else:
            deltas = S_t[:, None] - self.S[:, : step["n_starts"]]
            scores = np.abs(deltas) * step["inv_sqrt_len"]
            j_star = np.argmax(scores, axis=1)
            scan = scores[self.rows, j_star]
            delta = deltas[self.rows, j_star]

        self.scan_max = np.maximum(self.scan_max, scan)
        above = scan > THRESH
        self.first_cross3 = np.where((self.first_cross3 == 0) & above, t, self.first_cross3)
        self.above3_count += above
        since_cross3 = np.where(self.first_cross3 > 0, t - self.first_cross3 + 1, 0)

        dec_t = (self.C[:, :t] @ step["weights"]) * step["inv_w_norm"]
        full = S_t / math.sqrt(t)

        dec_list = dec_t.tolist()
        swing = []
        for i, d in enumerate(dec_list):
            hi, lo = self._swing_ref(i, step["memory"])
            if d > 0.0:
                swing.append(d / max(abs(hi), EPS))
            elif d < 0.0:
                swing.append(d / max(abs(lo), EPS))
            else:
                swing.append(0.0)

        return {
            "scan": scan.tolist(),
            "scan_max": self.scan_max.tolist(),
            "scan_dir": np.where(delta >= 0.0, 1.0, -1.0).tolist(),
            "scan_age": (t - j_star).astype(np.float64).tolist(),
            "cusum_up_max": self.cusum_up_max.tolist(),
            "cusum_dn_max": self.cusum_dn_max.tolist(),
            "dec_t": dec_list,
            "full": full.tolist(),
            "swing": swing,
            "since_cross3": since_cross3.astype(np.float64).tolist(),
        }


class LocalBank:
    """v3 group A: scan / full / t-scaled decay for signals calibrated on the recent history."""

    def __init__(self, n_signals, capacity=256):
        self.k = n_signals
        self.S = np.zeros((n_signals, capacity + 1), dtype=np.float64)
        self.C = np.zeros((n_signals, capacity), dtype=np.float64)
        self.scan_max = np.zeros(n_signals)

    def update(self, c, t, step):
        if t >= self.C.shape[1]:
            self.C = np.concatenate([self.C, np.zeros_like(self.C)], axis=1)
            self.S = np.concatenate([self.S, np.zeros((self.k, self.S.shape[1]))], axis=1)
        self.C[:, t - 1] = c
        S_t = self.S[:, t - 1] + c
        self.S[:, t] = S_t
        if t < MIN_SEG:
            scan = np.abs(S_t) / math.sqrt(t)
        else:
            scan = np.max(np.abs(S_t[:, None] - self.S[:, : step["n_starts"]]) * step["inv_sqrt_len"], axis=1)
        self.scan_max = np.maximum(self.scan_max, scan)
        dec_t = (self.C[:, :t] @ step["weights"]) * step["inv_w_norm"]
        return self.scan_max.tolist(), (S_t / math.sqrt(t)).tolist(), dec_t.tolist()


# ----------------------------- series features -----------------------------
class SeriesFeatures:
    """
    Causal feature generator: SeriesFeatures(history) once per series, then update(x_t) per point.
    Returns dict[str, float] in a fixed order; cast to float32 downstream.
    """

    def __init__(self, history):
        self.model = _history_model(history)
        self.t = 0
        self.bank = SignalBank(self.model["swing_refs"])
        self.local = LocalBank(len(LOCAL_SIGNALS))
        self.mu = np.asarray([self.model["calibrations"][n][0] for n in SIGNAL_NAMES])
        self.inv_sigma = 1.0 / np.maximum([self.model["calibrations"][n][1] for n in SIGNAL_NAMES], EPS)
        self.loc_idx = [_SIG_IDX[n] for n in LOCAL_SIGNALS]
        self.loc_mu = np.asarray([self.model["local_calibrations"][n][0] for n in LOCAL_SIGNALS])
        self.loc_inv_sigma = 1.0 / np.maximum([self.model["local_calibrations"][n][1] for n in LOCAL_SIGNALS], EPS)
        self.win_idx = [_SIG_IDX[n] for n in WINDOW_SIGNALS]
        self.small_idx = [_SIG_IDX[n] for n in SMALL_CUSUM_SIGNALS]
        self.small_up = np.zeros(len(SMALL_CUSUM_SIGNALS))
        self.small_dn = np.zeros(len(SMALL_CUSUM_SIGNALS))
        self.small_up_max = np.zeros(len(SMALL_CUSUM_SIGNALS))
        self.small_dn_max = np.zeros(len(SMALL_CUSUM_SIGNALS))
        self.best_scan_max_hist = deque(maxlen=TREND_LAG + 1)
        self.max_abs = 0.0
        self.n_beyond = 0
        self.last_beyond = None
        self.start_history = [deque(maxlen=LOC_WIN) for _ in SIGNAL_NAMES]
        self.context_items = list(self.model["context"].items())
        self.tail_items = list(self.model["tail_context"].items())

    def _raw_signals(self, x):
        beta = self.model["beta"]
        lags = self.model["lags"]
        prediction = beta[0] + sum(beta[i] * lags[i - 1] for i in range(1, AR_ORDER + 1))
        e = (x - prediction) / max(self.model["sigma_e"], EPS)
        z = float(self.model["z_table"][np.searchsorted(self.model["h_sorted"], x, side="right")])
        raw = np.asarray([
            x,                              # x
            e,                              # e
            z,                              # z
            e * e,                          # v
            z * z,                          # z2
            e * self.model["e_prev"],       # d
            x * x,                          # x2
            x * self.model["x_prev"],       # xx
        ], dtype=np.float64)
        return raw, e

    @staticmethod
    def _step_context(t):
        memory = max(5.0, T_SCALE * t)
        weights = np.exp(-np.arange(t - 1, -1, -1, dtype=np.float64) / memory)
        n_starts = max(t - MIN_SEG + 1, 0)
        return {
            "memory": memory,
            "weights": weights,
            "inv_w_norm": 1.0 / math.sqrt(max(float(np.dot(weights, weights)), EPS)),
            "n_starts": n_starts,
            "inv_sqrt_len": 1.0 / np.sqrt(t - np.arange(n_starts, dtype=np.float64)),
        }

    def update(self, x):
        x = float(x)
        self.t += 1
        t = self.t

        raw, e = self._raw_signals(x)
        c = (raw - self.mu) * self.inv_sigma
        c[~np.isfinite(c)] = 0.0
        step = self._step_context(t)
        out = self.bank.update(c, t, step)

        # ---------------- v1 features (unchanged) ----------------
        row = {}
        for i, name in enumerate(SIGNAL_NAMES):
            for feat_name in PER_SIGNAL_NAMES:
                row[f"{name}_{feat_name}"] = out[feat_name][i]
            self.start_history[i].append(t - out["scan_age"][i])

        for key, value in self.context_items:
            row[key] = value

        abs_x = abs(x)
        self.max_abs = max(self.max_abs, abs_x)
        if abs_x > self.model["hmax"]:
            self.n_beyond += 1
            self.last_beyond = t
        row["max_abs_ratio"] = self.max_abs / self.model["hmax"]
        row["n_beyond_hist_max"] = float(self.n_beyond)
        row["since_beyond"] = math.nan if self.last_beyond is None else float(t - self.last_beyond)

        scans = out["scan"]
        lead_idx = max(range(len(scans)), key=scans.__getitem__)
        row["lead_signal"] = float(lead_idx)
        row["best_scan"] = scans[lead_idx]
        row["best_scan_max"] = max(out["scan_max"])
        row["best_swing"] = max(abs(v) for v in out["swing"])
        row["n_sig_above3"] = float(sum(v > THRESH for v in scans))
        row["lead_frac_above3"] = float(self.bank.above3_count[lead_idx]) / t
        starts = self.start_history[lead_idx]
        if len(starts) >= 2:
            m = sum(starts) / len(starts)
            row["lead_loc_std"] = math.sqrt(sum((s - m) ** 2 for s in starts) / len(starts))
        else:
            row["lead_loc_std"] = 0.0
        row["log_t"] = math.log(t)

        # ---------------- v3 A: local-baseline detectors ----------------
        c_loc = (raw[self.loc_idx] - self.loc_mu) * self.loc_inv_sigma
        c_loc[~np.isfinite(c_loc)] = 0.0
        loc_scan_max, loc_full, loc_dec = self.local.update(c_loc, t, step)
        for i, name in enumerate(LOCAL_SIGNALS):
            row[f"loc_{name}_scan_max"] = loc_scan_max[i]
            row[f"loc_{name}_full"] = loc_full[i]
            row[f"loc_{name}_dec_t"] = loc_dec[i]

        # ---------------- v3 B: history-tail context ----------------
        for key, value in self.tail_items:
            row[key] = value

        # ---------------- v3 C: short online windows (whole-history calibration) ----------------
        S = self.bank.S
        for i, name in zip(self.win_idx, WINDOW_SIGNALS):
            for w in WINDOWS:
                wl = min(t, w)
                row[f"{name}_win{w}"] = (S[i, t] - S[i, t - wl]) / math.sqrt(wl)

        # ---------------- v3 D: small-shift CUSUM + evidence trend ----------------
        c_small = c[self.small_idx]
        self.small_up = np.maximum(0.0, self.small_up + c_small - SMALL_CUSUM_K)
        self.small_dn = np.maximum(0.0, self.small_dn - c_small - SMALL_CUSUM_K)
        self.small_up_max = np.maximum(self.small_up_max, self.small_up)
        self.small_dn_max = np.maximum(self.small_dn_max, self.small_dn)
        for i, name in enumerate(SMALL_CUSUM_SIGNALS):
            row[f"{name}_cusum_up_k25_max"] = float(self.small_up_max[i])
            row[f"{name}_cusum_dn_k25_max"] = float(self.small_dn_max[i])
        self.best_scan_max_hist.append(row["best_scan_max"])
        row[f"best_scan_max_d{TREND_LAG}"] = row["best_scan_max"] - self.best_scan_max_hist[0]

        # Advance lag state only AFTER all features for x_t are computed.
        self.model["lags"] = [x] + self.model["lags"][:-1]
        self.model["e_prev"] = float(e)
        self.model["x_prev"] = x
        return row


def feature_names():
    """Column order produced by SeriesFeatures.update (deterministic)."""
    rng = np.random.default_rng(0)
    gen = SeriesFeatures(rng.standard_normal(1200))
    return list(gen.update(0.0).keys())


# =====================================================================================
# 2. DATASET CLI (offline only)
# =====================================================================================
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import click
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

NON_FEATURE_COLS = [
    "id",
    "time",
    "online_step",
    "target",
    "tau_index",
    "tau",
    "has_break",
    "online_length",
]


def generate_online_features(history, online):
    generator = SeriesFeatures(history)
    rows = [generator.update(x) for x in np.asarray(online, dtype=np.float64)]
    return pd.DataFrame(rows).astype(np.float32)


def extract_series_target(series_y, online_times, series_id):
    if isinstance(series_y, pd.DataFrame):
        if "target" in series_y.columns:
            target = series_y["target"]
        elif "y" in series_y.columns:
            target = series_y["y"]
        elif len(series_y.columns) == 1:
            target = series_y.iloc[:, 0]
        else:
            raise ValueError(f"Could not determine target column for {series_id}.")
    else:
        target = series_y

    target = target.reindex(online_times)
    if target.isna().any():
        raise ValueError(f"Missing targets for series {series_id}.")
    return target.to_numpy()


def _tau_index_value(value):
    """-1 for 'no break' whether stored as -1, NaN or None."""
    if value is None or pd.isna(value):
        return -1
    return int(value)


def process_single_series(args):
    series_id, series, series_y, series_meta = args
    series = series.sort_index()
    history_df = series.loc[series["period"] == 1]
    online_df = series.loc[series["period"] == 2]
    if history_df.empty or online_df.empty:
        return None

    history = history_df["value"].to_numpy(dtype=np.float64)
    online = online_df["value"].to_numpy(dtype=np.float64)
    online_times = online_df.index.to_numpy()
    online_features = generate_online_features(history, online)
    targets = extract_series_target(series_y, online_times, series_id)
    n_online = len(online)

    if len(targets) != n_online:
        raise ValueError(f"{series_id}: {len(targets)} targets for {n_online} online rows.")
    if np.any(np.diff(np.asarray(targets, dtype=np.int64)) < 0):
        raise ValueError(f"{series_id}: target goes from 1 back to 0.")

    tau_index = _tau_index_value(series_meta["tau_index"])
    metadata = pd.DataFrame(
        {
            "id": np.repeat(series_id, n_online),
            "time": online_times,
            "online_step": np.arange(n_online, dtype=np.int32),
            "target": targets,
            "tau_index": np.full(n_online, tau_index, dtype=np.int32),
            "tau": np.repeat(series_meta["tau"], n_online),
            "has_break": np.full(n_online, int(tau_index >= 0), dtype=np.int8),
            "online_length": np.full(n_online, n_online, dtype=np.int32),
        }
    )
    return pd.concat([metadata, online_features], axis=1)


def read_dataframe(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def prepare_x_dataframe(X):
    if not isinstance(X.index, pd.MultiIndex):
        if not {"id", "time"}.issubset(X.columns):
            raise ValueError("X requires id/time columns or MultiIndex.")
        X = X.set_index(["id", "time"])
    if "id" not in X.index.names or "time" not in X.index.names:
        raise ValueError("X MultiIndex must contain id and time.")
    missing = {"value", "period"} - set(X.columns)
    if missing:
        raise ValueError(f"X missing columns: {sorted(missing)}")
    return X.sort_index()


def prepare_y_dataframe(y):
    if not isinstance(y.index, pd.MultiIndex):
        if not {"id", "time"}.issubset(y.columns):
            raise ValueError("y requires id/time columns or MultiIndex.")
        y = y.set_index(["id", "time"])
    if "id" not in y.index.names or "time" not in y.index.names:
        raise ValueError("y MultiIndex must contain id and time.")
    return y.sort_index()


def prepare_y_index_dataframe(y_index):
    if "id" in y_index.columns:
        y_index = y_index.set_index("id")
    if y_index.index.name != "id":
        raise ValueError("y_index requires id.")
    missing = {"tau_index", "tau"} - set(y_index.columns)
    if missing:
        raise ValueError(f"y_index missing: {sorted(missing)}")
    if not y_index.index.is_unique:
        raise ValueError("y_index must contain one row per id.")
    return y_index.sort_index()


def write_buffer(buffer, writer, output_path):
    if not buffer:
        return writer, 0
    chunk = pd.concat(buffer, ignore_index=True)
    table = pa.Table.from_pandas(chunk, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(str(output_path), table.schema, compression="zstd")
    writer.write_table(table)
    return writer, len(chunk)


def write_manifest(output_path, n_series, n_rows):
    """Record exactly which code and columns produced this file."""
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "feature_names": feature_names(),
        "non_feature_cols": NON_FEATURE_COLS,
        "n_series": n_series,
        "n_rows": n_rows,
    }
    path = Path(output_path).with_suffix(".manifest.json")
    path.write_text(json.dumps(manifest, indent=2))
    return path


def generate_dataset(X, y, y_index, output_path, num_workers=6, series_per_chunk=50):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() not in {".parquet", ".pq"}:
        raise ValueError("output_path must be a Parquet file.")

    series_ids = X.index.get_level_values("id").unique().tolist()
    missing_meta = [sid for sid in series_ids if sid not in y_index.index]
    if missing_meta:
        raise ValueError(f"{len(missing_meta)} series missing from y_index.")

    click.echo(f"Preparing {len(series_ids):,} series...")
    x_groups = {sid: g.droplevel("id") for sid, g in X.groupby(level="id", sort=False)}
    y_groups = {sid: g.droplevel("id") for sid, g in y.groupby(level="id", sort=False)}
    missing_y = [sid for sid in series_ids if sid not in y_groups]
    if missing_y:
        raise ValueError(f"{len(missing_y)} series missing from y.")

    def task_generator():
        for sid in series_ids:
            yield sid, x_groups[sid], y_groups[sid], y_index.loc[sid]

    click.echo(f"Generating features with {num_workers} workers...")
    buffer, writer = [], None
    total_rows = row_groups = 0
    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(process_single_series, task_generator(), chunksize=2)
            for result in tqdm(results, total=len(series_ids), desc="CPD features v3"):
                if result is None:
                    continue
                buffer.append(result)
                if len(buffer) < series_per_chunk:
                    continue
                writer, rows = write_buffer(buffer, writer, output_path)
                total_rows += rows
                row_groups += 1
                buffer.clear()
        if buffer:
            writer, rows = write_buffer(buffer, writer, output_path)
            total_rows += rows
            row_groups += 1
    finally:
        if writer is not None:
            writer.close()

    manifest_path = write_manifest(output_path, len(series_ids), total_rows)
    click.echo(f"Rows written: {total_rows:,}")
    click.echo(f"Row groups: {row_groups:,}")
    click.echo(f"Output: {output_path}")
    click.echo(f"Manifest: {manifest_path}")


@click.command()
@click.option("--x-path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--y-path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--y-index-path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--output-path", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--num-workers", type=click.IntRange(min=1), default=6, show_default=True)
@click.option("--series-per-chunk", type=click.IntRange(min=1), default=50, show_default=True)
def main(x_path, y_path, y_index_path, output_path, num_workers, series_per_chunk):
    click.echo(f"Reading X: {x_path}")
    X = prepare_x_dataframe(read_dataframe(x_path))
    click.echo(f"Reading y: {y_path}")
    y = prepare_y_dataframe(read_dataframe(y_path))
    click.echo(f"Reading y_index: {y_index_path}")
    y_index = prepare_y_index_dataframe(read_dataframe(y_index_path))

    names = feature_names()
    n_v1 = len(SIGNAL_NAMES) * len(PER_SIGNAL_NAMES) + 19
    click.echo(f"Series: {X.index.get_level_values('id').nunique():,}")
    click.echo(f"Workers: {num_workers}")
    click.echo(f"Series per chunk: {series_per_chunk}")
    click.echo(f"Model features: {len(names)} (v1 {n_v1} + v3 {len(names) - n_v1})")

    generate_dataset(
        X=X,
        y=y,
        y_index=y_index,
        output_path=output_path,
        num_workers=num_workers,
        series_per_chunk=series_per_chunk,
    )
    click.echo("Done.")


if __name__ == "__main__":
    main()
