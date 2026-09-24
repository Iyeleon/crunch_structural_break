from pathlib import Path

import click
import numpy as np
import pandas as pd

from tqdm.auto import tqdm


# Data loading

def load_old_data(data_dir):
    data_dir = Path(data_dir)

    x_train = pd.read_parquet(data_dir / "X_train.parquet")
    x_test = pd.read_parquet(data_dir / "X_test.reduced.parquet")
    y_train = pd.read_parquet(data_dir / "y_train.parquet")
    y_test = pd.read_parquet(data_dir / "y_test.reduced.parquet")

    # Remap test IDs so they do not overlap with train IDs
    train_ids = x_train.index.get_level_values("id")
    test_ids = x_test.index.get_level_values("id")

    id_offset = train_ids.max() + 1
    id_map = {
        old_id: id_offset + i
        for i, old_id in enumerate(test_ids.unique())
    }

    # Update X test IDs
    x_test = x_test.reset_index()
    x_test["id"] = x_test["id"].map(id_map)
    x_test = x_test.set_index(["id", "time"])

    # Update y test IDs
    y_test = y_test.reset_index()
    y_test["id"] = y_test["id"].map(id_map)

    index_cols = list(y_train.index.names)
    y_test = y_test.set_index(index_cols)

    # Combine train and test
    x = pd.concat([x_train, x_test]).sort_index()
    y = pd.concat([y_train, y_test]).sort_index()

    return x, y


# Series utilities

def get_boundary(df):
    periods = df["period"].to_numpy()
    indices = np.flatnonzero(periods == 1)

    if len(indices) == 0:
        return None

    return int(indices[0])


def get_target(y_train, series_id):
    target = y_train.loc[series_id]

    if isinstance(target, pd.Series):
        if "value" in target.index:
            target = target["value"]
        else:
            target = target.iloc[0]

    return bool(target)


def normalize_from_history(history, online, eps=1e-8):
    mean = history.mean()
    std = history.std()

    history = (history - mean) / (std + eps)
    online = (online - mean) / (std + eps)

    return history, online


# Sampling

def sample_online_length(rng, min_online, max_online):
    return int(rng.integers(min_online, max_online + 1))


def sample_history_length(rng, available, min_history, max_history):
    upper = min(available, max_history)

    if upper < min_history:
        return None

    return int(rng.integers(min_history, upper + 1))


def sample_positive_window(
    rng,
    values,
    boundary,
    min_history,
    min_online,
    max_online,
):
    online_length = sample_online_length(
        rng,
        min_online,
        min(max_online, len(values)),
    )

    max_break_position = min(
        online_length - 1,
        boundary - min_history,
    )

    if max_break_position < 1:
        return None

    break_position = int(
        rng.integers(1, max_break_position + 1)
    )

    online_start = boundary - break_position
    online_end = online_start + online_length

    if online_end > len(values):
        return None

    return online_start, online_end, break_position


def sample_negative_window(
    rng,
    values,
    boundary,
    min_history,
    min_online,
    max_online,
):
    max_online_length = min(
        max_online,
        boundary - min_history,
    )

    if max_online_length < min_online:
        return None

    online_length = sample_online_length(
        rng,
        min_online,
        max_online_length,
    )

    latest_start = boundary - online_length

    if latest_start < min_history:
        return None

    online_start = int(
        rng.integers(min_history, latest_start + 1)
    )

    online_end = online_start + online_length

    return online_start, online_end, None


# Sample construction

def build_sample(
    values,
    online_start,
    online_end,
    break_position,
    history_length,
    sample_id,
):
    history_start = online_start - history_length

    history = values[history_start:online_start].astype(
        np.float64,
        copy=True,
    )
    online = values[online_start:online_end].astype(
        np.float64,
        copy=True,
    )

    history, online = normalize_from_history(
        history,
        online,
    )

    n_history = len(history)
    n_online = len(online)

    combined = np.concatenate([history, online])

    periods = np.concatenate([
        np.ones(n_history, dtype=np.int8),
        np.full(n_online, 2, dtype=np.int8),
    ])

    x_sample = pd.DataFrame({
        "id": sample_id,
        "time": np.arange(len(combined)),
        "value": combined,
        "period": periods,
    })

    if break_position is None:
        targets = np.zeros(n_online, dtype=np.int8)
        tau_index = -1
        tau = -1

    else:
        targets = (
            np.arange(n_online) >= break_position
        ).astype(np.int8)

        tau_index = break_position
        tau = n_history + break_position

    y_sample = pd.DataFrame({
        "id": sample_id,
        "time": np.arange(
            n_history,
            n_history + n_online,
        ),
        "target": targets,
    })

    y_index = {
        "id": sample_id,
        "tau_index": tau_index,
        "tau": tau,
    }

    return x_sample, y_sample, y_index


def get_eligible_series(
    x_old,
    y_old,
    min_history,
):
    eligible = []

    groups = x_old.groupby(level="id")
    n_series = x_old.index.get_level_values("id").nunique()

    for series_id, df in tqdm(
        groups,
        total=n_series,
        desc="Finding eligible series",
    ):
        df = (
            df.reset_index(level="id", drop=True)
            .sort_index()
        )

        boundary = get_boundary(df)

        if boundary is None or boundary <= min_history:
            continue

        eligible.append({
            "id": series_id,
            "boundary": boundary,
            "target": get_target(y_old, series_id),
        })

    return pd.DataFrame(eligible)


def sample_negative_ids(negative_ids, target_count, rng):
    negative_ids = np.asarray(negative_ids)

    if len(negative_ids) >= target_count:
        return rng.choice(
            negative_ids,
            size=target_count,
            replace=False,
        )

    extra = rng.choice(
        negative_ids,
        size=target_count - len(negative_ids),
        replace=True,
    )

    return np.concatenate([
        negative_ids,
        extra,
    ])


def generate_one_sample(
    values,
    boundary,
    target,
    rng,
    sample_id,
    min_history,
    max_history,
    min_online,
    max_online,
):
    if target:
        window = sample_positive_window(
            rng=rng,
            values=values,
            boundary=boundary,
            min_history=min_history,
            min_online=min_online,
            max_online=max_online,
        )
    else:
        window = sample_negative_window(
            rng=rng,
            values=values,
            boundary=boundary,
            min_history=min_history,
            min_online=min_online,
            max_online=max_online,
        )

    if window is None:
        return None

    online_start, online_end, break_position = window

    history_length = sample_history_length(
        rng=rng,
        available=online_start,
        min_history=min_history,
        max_history=max_history,
    )

    if history_length is None:
        return None

    # Historical window ends exactly where online begins
    history_start = online_start - history_length

    if history_start < 0:
        return None

    return build_sample(
        values=values,
        online_start=online_start,
        online_end=online_end,
        break_position=break_position,
        history_length=history_length,
        sample_id=sample_id,
    )


def generate_augmented_dataset(
    x_old,
    y_old,
    samples_per_positive=3,
    min_history=1000,
    max_history=5000,
    min_online=10,
    max_online=1000,
    seed=42,
):
    rng = np.random.default_rng(seed)

    # Find eligible source series
    eligible = get_eligible_series(
        x_old=x_old,
        y_old=y_old,
        min_history=min_history,
    )

    positive = eligible[eligible["target"]]
    negative = eligible[~eligible["target"]]

    print(f"Eligible positive: {len(positive):,}")
    print(f"Eligible negative: {len(negative):,}")

    selected_ids = set(eligible["id"])

    series_cache = {
        series_id: (
            df.reset_index(level="id", drop=True)
            .sort_index()
        )
        for series_id, df in x_old.groupby(level="id")
        if series_id in selected_ids
    }

    x_samples = []
    y_samples = []
    y_indices = []

    sample_id = 0

    # Generate positive augmentations
    for row in tqdm(
        positive.itertuples(index=False),
        total=len(positive),
        desc="Generating positives",
    ):
        generated = 0
        attempts = 0

        values = series_cache[row.id]["value"].to_numpy()

        while (
            generated < samples_per_positive
            and attempts < samples_per_positive * 20
        ):
            attempts += 1

            sample = generate_one_sample(
                values=values,
                boundary=row.boundary,
                target=True,
                rng=rng,
                sample_id=sample_id,
                min_history=min_history,
                max_history=max_history,
                min_online=min_online,
                max_online=max_online,
            )

            if sample is None:
                continue

            x_sample, y_sample, y_index = sample

            x_samples.append(x_sample)
            y_samples.append(y_sample)
            y_indices.append(y_index)

            sample_id += 1
            generated += 1

    n_positive = sample_id

    # Select enough negative sources to match positives
    negative_ids = sample_negative_ids(
        negative["id"].to_numpy(),
        target_count=n_positive,
        rng=rng,
    )

    negative_lookup = negative.set_index("id")

    # Generate negative samples
    for series_id in tqdm(
        negative_ids,
        desc="Generating negatives",
    ):
        row = negative_lookup.loc[series_id]
        values = series_cache[series_id]["value"].to_numpy()

        sample = None

        for _ in range(20):
            sample = generate_one_sample(
                values=values,
                boundary=row["boundary"],
                target=False,
                rng=rng,
                sample_id=sample_id,
                min_history=min_history,
                max_history=max_history,
                min_online=min_online,
                max_online=max_online,
            )

            if sample is not None:
                break

        if sample is None:
            continue

        x_sample, y_sample, y_index = sample

        x_samples.append(x_sample)
        y_samples.append(y_sample)
        y_indices.append(y_index)

        sample_id += 1

    # Combine outputs
    x_new = (
        pd.concat(x_samples, ignore_index=True)
        .set_index(["id", "time"])
        .sort_index()
    )

    y_new = (
        pd.concat(y_samples, ignore_index=True)
        .set_index(["id", "time"])
        .sort_index()
    )

    y_index_new = (
        pd.DataFrame(y_indices)
        .set_index("id")
        .sort_index()
    )

    n_negative = len(y_index_new) - n_positive

    print(f"Positive samples: {n_positive:,}")
    print(f"Negative samples: {n_negative:,}")
    print(f"Total samples: {len(y_index_new):,}")

    return x_new, y_new, y_index_new


# Saving

def save_dataset(x, y, y_index, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x.to_parquet(output_dir / "X_train.parquet")
    y.to_parquet(output_dir / "y_train.parquet")
    y_index.to_parquet(output_dir / "y_index.parquet")


# CLI

@click.command()
@click.option(
    "--data-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
)
@click.option("--samples-per-positives", default=10, type=int)
@click.option("--min-history", default=1000, type=int)
@click.option("--max-history", default=5000, type=int)
@click.option("--min-online", default=10, type=int)
@click.option("--max-online", default=1000, type=int)
@click.option("--seed", default=42, type=int)
def main(
    data_dir,
    output_dir,
    samples_per_positives,
    min_history,
    max_history,
    min_online,
    max_online,
    seed,
):
    x_old, y_old = load_old_data(data_dir)

    x_new, y_new, y_index = generate_augmented_dataset(
        x_old=x_old,
        y_old=y_old,
        samples_per_positive=samples_per_positives,
        min_history=min_history,
        max_history=max_history,
        min_online=min_online,
        max_online=max_online,
        seed=seed,
    )

    save_dataset(
        x=x_new,
        y=y_new,
        y_index=y_index,
        output_dir=output_dir,
    )

    click.echo(f"Generated {len(y_index):,} series")
    click.echo(f"X observations: {len(x_new):,}")
    click.echo(f"Online observations: {len(y_new):,}")


if __name__ == "__main__":
    main()