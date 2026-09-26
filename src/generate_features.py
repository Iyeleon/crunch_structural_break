import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import click
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.special import ndtri
from tqdm.auto import tqdm


EPS = 1e-12
AR_ORDER = 3
CUSUM_K = 0.5
MIN_SEG = 5
THRESH = 3.0
M_FAST = 15
T_SCALE = 0.25
M_GRID = np.asarray([5, 10, 25, 50, 100, 250], dtype=np.float64)
SWING_Q = (0.005, 0.995)
LOC_WIN = 20

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
    m4 = np.mean(centered ** 4)
    value = m4 / (m2 * m2)
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
        [np.ones(y.size)]
        + [history[order - lag : history.size - lag] for lag in range(1, order + 1)]
    )
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    sigma_e = _safe_std(residuals)
    e_hist = residuals / sigma_e
    return beta, residuals, sigma_e, e_hist


def _right_rank_u(values, sorted_history):
    values = np.asarray(values, dtype=np.float64)
    n = len(sorted_history)
    ranks = np.searchsorted(sorted_history, values, side="right")
    u = (ranks.astype(np.float64) + 0.5) / (n + 1.0)
    return np.clip(u, 1e-6, 1.0 - 1e-6)


def _newey_west_std(x):
    """Bartlett-kernel long-run std with the requested automatic bandwidth."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return 1.0

    xc = x - np.mean(x)
    gamma0 = float(np.mean(xc * xc))
    if not np.isfinite(gamma0) or gamma0 <= EPS:
        return np.sqrt(EPS)

    bandwidth = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    bandwidth = min(max(bandwidth, 0), n - 1)

    lrv = gamma0
    for lag in range(1, bandwidth + 1):
        gamma = float(np.mean(xc[lag:] * xc[:-lag]))
        weight = 1.0 - lag / (bandwidth + 1.0)
        lrv += 2.0 * weight * gamma

    floor = max(0.1 * gamma0, EPS)
    lrv = max(lrv, floor) if np.isfinite(lrv) else floor
    return float(np.sqrt(lrv))


def _fixed_decay_series(c_hist, memory):
    a = np.exp(-1.0 / float(memory))
    A = 0.0
    B = 0.0
    out = np.empty(len(c_hist), dtype=np.float64)
    for i, c in enumerate(c_hist):
        A = a * A + c
        B = (a * a) * B + 1.0
        out[i] = A / np.sqrt(max(B, EPS))
    return out


def _build_swing_reference(c_hist):
    refs_m, refs_hi, refs_lo = [], [], []
    for memory in M_GRID:
        values = _fixed_decay_series(c_hist, memory)
        burn = int(3 * memory)
        if values.size <= burn:
            continue
        values = values[burn:]
        lo, hi = np.quantile(values, SWING_Q)
        if np.isfinite(lo) and np.isfinite(hi):
            refs_m.append(float(memory))
            refs_hi.append(float(hi))
            refs_lo.append(float(lo))

    # Short histories can make every 3m burn-in invalid. Fall back to the
    # smallest-memory full series so swing remains defined and deterministic.
    if not refs_m:
        memory = float(M_GRID[0])
        values = _fixed_decay_series(c_hist, memory)
        lo, hi = np.quantile(values, SWING_Q)
        refs_m = [memory]
        refs_hi = [float(hi)]
        refs_lo = [float(lo)]

    return (
        np.asarray(refs_m, dtype=np.float64),
        np.asarray(refs_hi, dtype=np.float64),
        np.asarray(refs_lo, dtype=np.float64),
    )


def _interp_swing_reference(reference, memory):
    m, hi, lo = reference
    if m.size == 1:
        return float(hi[0]), float(lo[0])
    log_m = np.log(m)
    query = np.log(np.clip(float(memory), m[0], m[-1]))
    return (
        float(np.interp(query, log_m, hi)),
        float(np.interp(query, log_m, lo)),
    )


def _history_model(history):
    history = np.asarray(history, dtype=np.float64)
    beta, residuals, sigma_e, e_hist = _fit_ar(history)
    h_sorted = np.sort(history)

    h_core = history[AR_ORDER:]
    z_hist = ndtri(_right_rank_u(h_core, h_sorted))
    e_prev_hist = np.r_[0.0, e_hist[:-1]]
    x_prev_hist = history[AR_ORDER - 1 : -1]

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

    calibrations = {}
    swing_refs = {}
    for name in SIGNAL_NAMES:
        values = np.asarray(raw_hist[name], dtype=np.float64)
        mu = float(np.mean(values))
        sigma = _newey_west_std(values)
        c_hist = (values - mu) / max(sigma, EPS)
        c_hist[~np.isfinite(c_hist)] = 0.0
        calibrations[name] = (mu, sigma)
        swing_refs[name] = _build_swing_reference(c_hist)

    abs_h = np.abs(history)
    q99 = float(np.quantile(abs_h, 0.99))
    hmax = max(float(np.max(abs_h)), EPS)
    hist_std = _safe_std(history)

    context = {
        "hist_length": float(len(history)),
        "ar1": float(beta[1]),
        "ar_sum": float(np.sum(beta[1:])),
        "resid_ratio": float(sigma_e / hist_std),
        "kurt": _kurtosis_nonexcess(e_hist),
        "arch1": _corr1(e_hist ** 2),
        "hist_q99": q99,
        "hist_max_abs": hmax,
    }

    return {
        "beta": beta,
        "sigma_e": sigma_e,
        "h_sorted": h_sorted,
        "q99": q99,
        "hmax": hmax,
        "lags": list(history[-AR_ORDER:][::-1]),  # most recent first
        "e_prev": float(e_hist[-1]),
        "x_prev": float(history[-1]),
        "calibrations": calibrations,
        "swing_refs": swing_refs,
        "context": context,
    }


class SignalState:
    def __init__(self, swing_reference):
        self.S_values = [0.0]  # S_0, S_1, ...
        self.c_values = []
        self.cusum_up = 0.0
        self.cusum_dn = 0.0
        self.cusum_up_max = 0.0
        self.cusum_dn_max = 0.0
        self.scan_max = 0.0
        self.first_cross3 = None
        self.swing_reference = swing_reference
        self.current_scan = 0.0
        self.current_start = 0

    def update(self, c, t):
        c = float(c) if np.isfinite(c) else 0.0
        self.c_values.append(c)
        S_t = self.S_values[-1] + c
        self.S_values.append(S_t)

        # CUSUM and their persistent maxima.
        self.cusum_up = max(0.0, self.cusum_up + c - CUSUM_K)
        self.cusum_dn = max(0.0, self.cusum_dn - c - CUSUM_K)
        self.cusum_up_max = max(self.cusum_up_max, self.cusum_up)
        self.cusum_dn_max = max(self.cusum_dn_max, self.cusum_dn)

        # Current causal scan. j indexes the cumulative sum before the segment.
        if t < MIN_SEG:
            scan = abs(S_t) / np.sqrt(float(t))
            j_star = 0
            delta = S_t
        else:
            j = np.arange(0, t - MIN_SEG + 1, dtype=np.int64)
            S_arr = np.asarray(self.S_values, dtype=np.float64)
            deltas = S_t - S_arr[j]
            ages = t - j
            scores = np.abs(deltas) / np.sqrt(ages)
            best = int(np.argmax(scores))
            scan = float(scores[best])
            j_star = int(j[best])
            delta = float(deltas[best])

        scan_dir = 1.0 if delta >= 0.0 else -1.0
        scan_age = float(t - j_star)
        self.current_scan = scan
        self.current_start = j_star
        self.scan_max = max(self.scan_max, scan)
        if self.first_cross3 is None and scan > THRESH:
            self.first_cross3 = t
        since_cross3 = 0.0 if self.first_cross3 is None else float(t - self.first_cross3 + 1)

        # t-scaled exponential evidence, recomputed exactly from all online c's.
        memory = max(5.0, T_SCALE * t)
        values = np.asarray(self.c_values, dtype=np.float64)
        ages = np.arange(t - 1, -1, -1, dtype=np.float64)
        weights = np.exp(-ages / memory)
        dec_t = float(np.dot(weights, values) / np.sqrt(max(np.dot(weights, weights), EPS)))
        full = float(S_t / np.sqrt(float(t)))

        hi, lo = _interp_swing_reference(self.swing_reference, memory)
        if dec_t > 0.0:
            swing = dec_t / max(abs(hi), EPS)
        elif dec_t < 0.0:
            swing = dec_t / max(abs(lo), EPS)
        else:
            swing = 0.0

        return {
            "scan": scan,
            "scan_max": self.scan_max,
            "scan_dir": scan_dir,
            "scan_age": scan_age,
            "cusum_up_max": self.cusum_up_max,
            "cusum_dn_max": self.cusum_dn_max,
            "dec_t": dec_t,
            "full": full,
            "swing": swing,
            "since_cross3": since_cross3,
        }


class SeriesFeatures:
    """Causal feature generator: call update(x_t) once for each online point."""

    def __init__(self, history):
        self.model = _history_model(history)
        self.t = 0
        self.states = {
            name: SignalState(self.model["swing_refs"][name])
            for name in SIGNAL_NAMES
        }
        self.max_abs = 0.0
        self.n_beyond = 0
        self.last_beyond = None
        self.start_history = {name: [] for name in SIGNAL_NAMES}

    def _raw_signals(self, x):
        beta = self.model["beta"]
        lags = self.model["lags"]
        prediction = beta[0] + sum(beta[i] * lags[i - 1] for i in range(1, AR_ORDER + 1))
        e = (x - prediction) / max(self.model["sigma_e"], EPS)
        u = float(_right_rank_u(np.asarray([x]), self.model["h_sorted"])[0])
        z = float(ndtri(u))
        return {
            "x": x,
            "e": e,
            "z": z,
            "v": e * e,
            "z2": z * z,
            "d": e * self.model["e_prev"],
            "x2": x * x,
            "xx": x * self.model["x_prev"],
        }, e

    def update(self, x):
        x = float(x)
        self.t += 1
        t = self.t

        raw, e = self._raw_signals(x)
        row = {}
        current_scans = {}
        current_swings = {}

        for name in SIGNAL_NAMES:
            mu, sigma = self.model["calibrations"][name]
            c = (raw[name] - mu) / max(sigma, EPS)
            if not np.isfinite(c):
                c = 0.0
            feats = self.states[name].update(c, t)
            current_scans[name] = feats["scan"]
            current_swings[name] = feats["swing"]
            self.start_history[name].append(float(t - feats["scan_age"]))
            for feat_name in PER_SIGNAL_NAMES:
                row[f"{name}_{feat_name}"] = feats[feat_name]

        # Static historical context.
        row.update(self.model["context"])

        # Global extremes.
        abs_x = abs(x)
        self.max_abs = max(self.max_abs, abs_x)
        if abs_x > self.model["hmax"]:
            self.n_beyond += 1
            self.last_beyond = t
        row["max_abs_ratio"] = self.max_abs / self.model["hmax"]
        row["n_beyond_hist_max"] = float(self.n_beyond)
        row["since_beyond"] = np.nan if self.last_beyond is None else float(t - self.last_beyond)

        # Cross-signal summaries use CURRENT scan for leading-signal selection.
        lead_idx = int(np.argmax([current_scans[name] for name in SIGNAL_NAMES]))
        lead_name = SIGNAL_NAMES[lead_idx]
        row["lead_signal"] = float(lead_idx)
        row["best_scan"] = max(current_scans.values())
        row["best_scan_max"] = max(self.states[name].scan_max for name in SIGNAL_NAMES)
        row["best_swing"] = max(abs(v) for v in current_swings.values())
        row["n_sig_above3"] = float(sum(v > THRESH for v in current_scans.values()))

        lead_scans = np.asarray([self.states[lead_name].current_scan], dtype=np.float64)
        # Fraction is maintained from the actual scan history; derive it from a counter
        # stored lazily on the state to avoid keeping another output feature per signal.
        state = self.states[lead_name]
        if not hasattr(state, "above3_count"):
            state.above3_count = 0
        # Every state needs its counter updated, not only the current leader.
        for name in SIGNAL_NAMES:
            s = self.states[name]
            if not hasattr(s, "above3_count"):
                s.above3_count = 0
            if s.current_scan > THRESH:
                s.above3_count += 1
        row["lead_frac_above3"] = float(state.above3_count / t)

        starts = self.start_history[lead_name][-LOC_WIN:]
        row["lead_loc_std"] = float(np.std(starts, ddof=0)) if len(starts) >= 2 else 0.0
        row["log_t"] = float(np.log(t))

        # Update causal lag state only AFTER all features for x_t are computed.
        self.model["lags"] = [x] + self.model["lags"][:-1]
        self.model["e_prev"] = float(e)
        self.model["x_prev"] = x

        # Fixed order and float32 outputs.
        return {name: np.float32(value) for name, value in row.items()}


def generate_online_features(history, online):
    generator = SeriesFeatures(history)
    rows = [generator.update(x) for x in np.asarray(online, dtype=np.float64)]
    return pd.DataFrame(rows)


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

    tau_index = int(series_meta["tau_index"])
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
            for result in tqdm(results, total=len(series_ids), desc="CPD features"):
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

    click.echo(f"Rows written: {total_rows:,}")
    click.echo(f"Row groups: {row_groups:,}")
    click.echo(f"Output: {output_path}")


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

    click.echo(f"Series: {X.index.get_level_values('id').nunique():,}")
    click.echo(f"Workers: {num_workers}")
    click.echo(f"Series per chunk: {series_per_chunk}")
    click.echo("Model features: 99 (80 per-signal + 19 global)")

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
