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
from tqdm.auto import tqdm


RECENT_HIST_WINDOW = 50

HIST_LEVEL_LAGS = (
    list(range(1, 51))
    + [75, 100]
    + list(range(200, 1001, 100))
)

HIST_DIFF_LAGS = list(range(1, 51))
ONLINE_LAGS = list(range(1, 51))
ONLINE_WINDOWS = [5, 10, 20, 50]

META_COLS = [
    "id",
    "time",
    "online_step",
    "target",
    "tau_index",
    "tau",
    "has_break",
    "online_length",
]


def generate_historical_features(
    history,
    ema_span,
):
    history = np.asarray(
        history,
        dtype=np.float64,
    )

    history_series = pd.Series(
        history,
        dtype=np.float64,
    )

    smooth = (
        history_series
        .ewm(
            span=ema_span,
            adjust=False,
        )
        .mean()
    )

    diff1 = smooth.diff()
    diff2 = diff1.diff()

    features = {}

    for lag in HIST_LEVEL_LAGS:
        features[
            f"hist_smooth_lag_{lag}"
        ] = (
            smooth.iloc[-lag]
            if len(smooth) >= lag
            else np.nan
        )

    for lag in HIST_DIFF_LAGS:
        features[
            f"hist_diff1_lag_{lag}"
        ] = (
            diff1.iloc[-lag]
            if len(diff1) >= lag
            else np.nan
        )

        features[
            f"hist_diff2_lag_{lag}"
        ] = (
            diff2.iloc[-lag]
            if len(diff2) >= lag
            else np.nan
        )

    recent = history[
        -RECENT_HIST_WINDOW:
    ]

    features[
        "hist_recent50_mean"
    ] = np.mean(
        recent
    )

    features[
        "hist_recent50_std"
    ] = np.std(
        recent,
        ddof=0,
    )

    return features


def compute_online_rolling_features(
    full_series,
    online_start,
):
    features = {}

    for window in ONLINE_WINDOWS:
        rolling = (
            full_series
            .rolling(
                window=window,
                min_periods=1,
            )
        )

        features[
            f"online_w{window}_mean"
        ] = (
            rolling
            .mean()
            .iloc[
                online_start:
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        features[
            f"online_w{window}_std"
        ] = (
            rolling
            .std(
                ddof=0
            )
            .iloc[
                online_start:
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

    return features


def generate_online_features(
    history,
    online,
    ema_span,
):
    history = np.asarray(
        history,
        dtype=np.float64,
    )

    online = np.asarray(
        online,
        dtype=np.float64,
    )

    full_values = np.concatenate(
        [
            history,
            online,
        ]
    )

    full_series = pd.Series(
        full_values,
        dtype=np.float64,
    )

    online_start = len(
        history
    )

    smooth = (
        full_series
        .ewm(
            span=ema_span,
            adjust=False,
        )
        .mean()
    )

    diff1 = smooth.diff()
    diff2 = diff1.diff()

    features = {}

    for lag in ONLINE_LAGS:
        shift = lag - 1

        features[
            f"online_smooth_lag_{lag}"
        ] = (
            smooth
            .shift(shift)
            .iloc[
                online_start:
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        features[
            f"online_diff1_lag_{lag}"
        ] = (
            diff1
            .shift(shift)
            .iloc[
                online_start:
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        features[
            f"online_diff2_lag_{lag}"
        ] = (
            diff2
            .shift(shift)
            .iloc[
                online_start:
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

    rolling_features = (
        compute_online_rolling_features(
            full_series=full_series,
            online_start=online_start,
        )
    )

    features.update(
        rolling_features
    )

    return pd.DataFrame(
        features
    )


def extract_series_target(
    series_y,
    online_times,
    series_id,
):
    if isinstance(
        series_y,
        pd.DataFrame,
    ):
        if (
            "target"
            in series_y.columns
        ):
            target = (
                series_y[
                    "target"
                ]
            )

        elif (
            "y"
            in series_y.columns
        ):
            target = (
                series_y["y"]
            )

        elif (
            len(
                series_y.columns
            )
            == 1
        ):
            target = (
                series_y.iloc[
                    :,
                    0,
                ]
            )

        else:
            raise ValueError(
                f"Could not determine "
                f"target column for "
                f"{series_id}."
            )

    else:
        target = series_y

    target = (
        target.reindex(
            online_times
        )
    )

    if target.isna().any():
        raise ValueError(
            f"Missing targets for "
            f"series {series_id}."
        )

    return target.to_numpy()


def process_single_series(args):
    (
        series_id,
        series,
        series_y,
        series_meta,
        ema_span,
    ) = args

    series = (
        series.sort_index()
    )

    history_df = series.loc[
        series[
            "period"
        ]
        == 1
    ]

    online_df = series.loc[
        series[
            "period"
        ]
        == 2
    ]

    if (
        history_df.empty
        or online_df.empty
    ):
        return None

    history = (
        history_df[
            "value"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    online = (
        online_df[
            "value"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    online_times = (
        online_df.index
        .to_numpy()
    )

    historical_features = (
        generate_historical_features(
            history=history,
            ema_span=ema_span,
        )
    )

    online_features = (
        generate_online_features(
            history=history,
            online=online,
            ema_span=ema_span,
        )
    )

    targets = (
        extract_series_target(
            series_y=series_y,
            online_times=online_times,
            series_id=series_id,
        )
    )

    n_online = len(
        online
    )

    if (
        len(targets)
        != n_online
    ):
        raise ValueError(
            f"{series_id}: "
            f"{len(targets)} targets "
            f"for {n_online} "
            f"online rows."
        )

    historical_df = pd.DataFrame(
        {
            name: np.full(
                n_online,
                value,
                dtype=np.float32,
            )
            for (
                name,
                value,
            )
            in historical_features.items()
        }
    )

    tau_index = int(
        series_meta[
            "tau_index"
        ]
    )

    metadata = pd.DataFrame(
        {
            "id": np.repeat(
                series_id,
                n_online,
            ),
            "time": online_times,
            "online_step": np.arange(
                n_online,
                dtype=np.int32,
            ),
            "target": targets,
            "tau_index": np.full(
                n_online,
                tau_index,
                dtype=np.int32,
            ),
            "tau": np.repeat(
                series_meta[
                    "tau"
                ],
                n_online,
            ),
            "has_break": np.full(
                n_online,
                int(
                    tau_index >= 0
                ),
                dtype=np.int8,
            ),
            "online_length": np.full(
                n_online,
                n_online,
                dtype=np.int32,
            ),
        }
    )

    return pd.concat(
        [
            metadata,
            historical_df,
            online_features,
        ],
        axis=1,
    )


def read_dataframe(path):
    path = Path(
        path
    )

    suffix = (
        path.suffix.lower()
    )

    if suffix in {
        ".parquet",
        ".pq",
    }:
        return pd.read_parquet(
            path
        )

    if suffix == ".csv":
        return pd.read_csv(
            path
        )

    if suffix in {
        ".pkl",
        ".pickle",
    }:
        return pd.read_pickle(
            path
        )

    raise ValueError(
        f"Unsupported file "
        f"type: {suffix}"
    )


def prepare_x_dataframe(X):
    if not isinstance(
        X.index,
        pd.MultiIndex,
    ):
        required = {
            "id",
            "time",
        }

        if not required.issubset(
            X.columns
        ):
            raise ValueError(
                "X requires id/time "
                "columns or MultiIndex."
            )

        X = X.set_index(
            [
                "id",
                "time",
            ]
        )

    if (
        "id"
        not in X.index.names
        or "time"
        not in X.index.names
    ):
        raise ValueError(
            "X MultiIndex must "
            "contain id and time."
        )

    required_columns = {
        "value",
        "period",
    }

    missing = (
        required_columns
        - set(
            X.columns
        )
    )

    if missing:
        raise ValueError(
            f"X missing columns: "
            f"{sorted(missing)}"
        )

    return X.sort_index()


def prepare_y_dataframe(y):
    if not isinstance(
        y.index,
        pd.MultiIndex,
    ):
        required = {
            "id",
            "time",
        }

        if not required.issubset(
            y.columns
        ):
            raise ValueError(
                "y requires id/time "
                "columns or MultiIndex."
            )

        y = y.set_index(
            [
                "id",
                "time",
            ]
        )

    if (
        "id"
        not in y.index.names
        or "time"
        not in y.index.names
    ):
        raise ValueError(
            "y MultiIndex must "
            "contain id and time."
        )

    return y.sort_index()


def prepare_y_index_dataframe(
    y_index,
):
    if (
        "id"
        in y_index.columns
    ):
        y_index = (
            y_index.set_index(
                "id"
            )
        )

    if (
        y_index.index.name
        != "id"
    ):
        raise ValueError(
            "y_index requires id."
        )

    required = {
        "tau_index",
        "tau",
    }

    missing = (
        required
        - set(
            y_index.columns
        )
    )

    if missing:
        raise ValueError(
            f"y_index missing: "
            f"{sorted(missing)}"
        )

    if not (
        y_index.index.is_unique
    ):
        raise ValueError(
            "y_index must contain "
            "one row per id."
        )

    return y_index.sort_index()


def generate_dataset(
    X,
    y,
    y_index,
    output_path,
    ema_span=5,
    num_workers=6,
    series_per_chunk=50,
):
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        output_path.suffix.lower()
        not in {
            ".parquet",
            ".pq",
        }
    ):
        raise ValueError(
            "output_path must be "
            "a Parquet file."
        )

    series_ids = (
        X.index
        .get_level_values(
            "id"
        )
        .unique()
        .tolist()
    )

    missing_meta = [
        series_id
        for series_id in series_ids
        if series_id
        not in y_index.index
    ]

    if missing_meta:
        raise ValueError(
            f"{len(missing_meta)} "
            f"series missing from "
            f"y_index."
        )

    click.echo(
        f"Preparing "
        f"{len(series_ids):,} "
        f"series..."
    )

    x_groups = {
        series_id: (
            group.droplevel(
                "id"
            )
        )
        for (
            series_id,
            group,
        )
        in X.groupby(
            level="id",
            sort=False,
        )
    }

    y_groups = {
        series_id: (
            group.droplevel(
                "id"
            )
        )
        for (
            series_id,
            group,
        )
        in y.groupby(
            level="id",
            sort=False,
        )
    }

    missing_y = [
        series_id
        for series_id in series_ids
        if series_id
        not in y_groups
    ]

    if missing_y:
        raise ValueError(
            f"{len(missing_y)} "
            f"series missing from y."
        )

    def task_generator():
        for series_id in series_ids:
            yield (
                series_id,
                x_groups[
                    series_id
                ],
                y_groups[
                    series_id
                ],
                y_index.loc[
                    series_id
                ],
                ema_span,
            )

    click.echo(
        f"Generating features "
        f"with {num_workers} "
        f"workers..."
    )

    buffer = []
    writer = None

    total_rows = 0
    row_groups = 0

    try:
        with ProcessPoolExecutor(
            max_workers=num_workers,
        ) as executor:
            results = executor.map(
                process_single_series,
                task_generator(),
                chunksize=2,
            )

            for result in tqdm(
                results,
                total=len(
                    series_ids
                ),
                desc=(
                    f"EMA({ema_span})"
                ),
            ):
                if result is None:
                    continue

                buffer.append(
                    result
                )

                if (
                    len(buffer)
                    < series_per_chunk
                ):
                    continue

                chunk = pd.concat(
                    buffer,
                    ignore_index=True,
                )

                table = (
                    pa.Table.from_pandas(
                        chunk,
                        preserve_index=False,
                    )
                )

                if writer is None:
                    writer = (
                        pq.ParquetWriter(
                            str(
                                output_path
                            ),
                            table.schema,
                            compression="zstd",
                        )
                    )

                writer.write_table(
                    table
                )

                total_rows += len(
                    chunk
                )

                row_groups += 1

                buffer.clear()

                del chunk
                del table

        if buffer:
            chunk = pd.concat(
                buffer,
                ignore_index=True,
            )

            table = (
                pa.Table.from_pandas(
                    chunk,
                    preserve_index=False,
                )
            )

            if writer is None:
                writer = (
                    pq.ParquetWriter(
                        str(
                            output_path
                        ),
                        table.schema,
                        compression="zstd",
                    )
                )

            writer.write_table(
                table
            )

            total_rows += len(
                chunk
            )

            row_groups += 1

            buffer.clear()

            del chunk
            del table

    finally:
        if writer is not None:
            writer.close()

    click.echo("")

    click.echo(
        f"Rows written: "
        f"{total_rows:,}"
    )

    click.echo(
        f"Row groups: "
        f"{row_groups:,}"
    )

    click.echo(
        f"Output: "
        f"{output_path}"
    )


@click.command()
@click.option(
    "--x-path",
    type=click.Path(
        exists=True,
        dir_okay=False,
        path_type=Path,
    ),
    required=True,
)
@click.option(
    "--y-path",
    type=click.Path(
        exists=True,
        dir_okay=False,
        path_type=Path,
    ),
    required=True,
)
@click.option(
    "--y-index-path",
    type=click.Path(
        exists=True,
        dir_okay=False,
        path_type=Path,
    ),
    required=True,
)
@click.option(
    "--output-path",
    type=click.Path(
        dir_okay=False,
        path_type=Path,
    ),
    required=True,
)
@click.option(
    "--ema-span",
    type=click.IntRange(
        min=1
    ),
    default=5,
    show_default=True,
)
@click.option(
    "--num-workers",
    type=click.IntRange(
        min=1
    ),
    default=6,
    show_default=True,
)
@click.option(
    "--series-per-chunk",
    type=click.IntRange(
        min=1
    ),
    default=50,
    show_default=True,
)
def main(
    x_path,
    y_path,
    y_index_path,
    output_path,
    ema_span,
    num_workers,
    series_per_chunk,
):
    click.echo(
        f"Reading X: "
        f"{x_path}"
    )

    X = prepare_x_dataframe(
        read_dataframe(
            x_path
        )
    )

    click.echo(
        f"Reading y: "
        f"{y_path}"
    )

    y = prepare_y_dataframe(
        read_dataframe(
            y_path
        )
    )

    click.echo(
        f"Reading y_index: "
        f"{y_index_path}"
    )

    y_index = (
        prepare_y_index_dataframe(
            read_dataframe(
                y_index_path
            )
        )
    )

    n_series = (
        X.index
        .get_level_values(
            "id"
        )
        .nunique()
    )

    click.echo(
        f"Series: "
        f"{n_series:,}"
    )

    click.echo(
        f"EMA span: "
        f"{ema_span}"
    )

    click.echo(
        f"Workers: "
        f"{num_workers}"
    )

    click.echo(
        f"Series per chunk: "
        f"{series_per_chunk}"
    )

    click.echo(
        "Model features: 322"
    )

    generate_dataset(
        X=X,
        y=y,
        y_index=y_index,
        output_path=(
            output_path
        ),
        ema_span=(
            ema_span
        ),
        num_workers=(
            num_workers
        ),
        series_per_chunk=(
            series_per_chunk
        ),
    )

    click.echo(
        "Done."
    )


if __name__ == "__main__":
    main()