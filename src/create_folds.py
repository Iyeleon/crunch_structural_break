from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


# Split metadata

def build_split_metadata(x, y_index):
    lengths = (
        x.reset_index()
        .groupby(["id", "period"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={
            1: "history_length",
            2: "online_length",
        })
    )

    meta = lengths.join(
        y_index[["tau_index", "tau"]]
    )

    meta["relative_break"] = np.where(
        meta["tau_index"] == -1,
        0.0,
        meta["tau_index"] / meta["online_length"],
    )

    meta["relative_break_bin"] = pd.cut(
        meta["relative_break"],
        bins=[-0.01, 0, 0.25, 0.50, 0.75, 1.0],
        labels=[0, 1, 2, 3, 4],
    ).astype(int)

    meta["online_length_bin"] = pd.cut(
        meta["online_length"],
        bins=[9, 50, 100, 250, 500, 1000],
        labels=[0, 1, 2, 3, 4],
    ).astype(int)

    meta["strata"] = (
        meta["relative_break_bin"].astype(str)
        + "_"
        + meta["online_length_bin"].astype(str)
    )

    return meta


# Fold generation

def create_folds(meta, n_splits=5, seed=42):
    meta = meta.copy()
    meta["fold"] = -1

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    for fold, (_, val_idx) in enumerate(
        splitter.split(meta, meta["strata"])
    ):
        meta.iloc[
            val_idx,
            meta.columns.get_loc("fold"),
        ] = fold

    return meta


# Fold validation

def print_fold_summary(folds):
    print("\nBreak position")
    print(
        pd.crosstab(
            folds["fold"],
            folds["relative_break_bin"],
        )
    )

    print("\nOnline length")
    print(
        pd.crosstab(
            folds["fold"],
            folds["online_length_bin"],
        )
    )

    print("\nLengths")
    print(
        folds.groupby("fold")[
            ["history_length", "online_length"]
        ].agg([
            "count",
            "mean",
            "std",
            "min",
            "median",
            "max",
        ])
    )


# CLI

@click.command()
@click.option(
    "--x-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option(
    "--y-index-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option(
    "--output-path",
    type=click.Path(path_type=Path),
    required=True,
)
@click.option("--n-splits", default=5, type=int)
@click.option("--seed", default=42, type=int)
def main(
    x_path,
    y_index_path,
    output_path,
    n_splits,
    seed,
):
    x = pd.read_parquet(x_path)
    y_index = pd.read_parquet(y_index_path)

    meta = build_split_metadata(
        x=x,
        y_index=y_index,
    )

    print("\nStrata distribution")
    print(
        meta["strata"]
        .value_counts()
        .sort_index()
    )

    folds = create_folds(
        meta=meta,
        n_splits=n_splits,
        seed=seed,
    )

    print_fold_summary(folds)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    folds.to_parquet(output_path)

    print(f"\nSaved folds to: {output_path}")


if __name__ == "__main__":
    main()