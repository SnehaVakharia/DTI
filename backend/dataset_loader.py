from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tdc.multi_pred import DTI


STANDARD_COLUMNS = ["smiles", "protein_sequence", "affinity", "label"]
LOWER_IS_BETTER_DATASETS = {"DAVIS", "BindingDB_Kd"}
HIGHER_IS_BETTER_DATASETS = {"KIBA"}


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    threshold: float
    max_samples: int


def _to_standard_frame(raw: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Drug": "smiles",
        "Target": "protein_sequence",
        "Y": "affinity",
    }
    df = raw.rename(columns=rename_map).copy()
    missing = set(rename_map.values()) - set(df.columns)
    if missing:
        missing_txt = ", ".join(sorted(missing))
        raise ValueError(f"Dataset is missing required columns after rename: {missing_txt}")

    df = df[["smiles", "protein_sequence", "affinity"]].copy()
    df["smiles"] = df["smiles"].astype(str).str.replace(r"\s+", "", regex=True)
    df["protein_sequence"] = df["protein_sequence"].astype(str).str.replace(r"\s+", "", regex=True).str.upper()
    df["affinity"] = pd.to_numeric(df["affinity"], errors="coerce")
    df = df.dropna(subset=["smiles", "protein_sequence", "affinity"]).reset_index(drop=True)
    return df


def _binarize_labels(df: pd.DataFrame, dataset_name: str, threshold: float) -> pd.DataFrame:
    out = df.copy()
    if dataset_name in LOWER_IS_BETTER_DATASETS:
        out["label"] = (out["affinity"] <= threshold).astype(int)
    elif dataset_name in HIGHER_IS_BETTER_DATASETS:
        out["label"] = (out["affinity"] >= threshold).astype(int)
    else:
        raise ValueError(f"Unsupported dataset for binarization: {dataset_name}")
    return out


def _stratified_sample(df: pd.DataFrame, max_samples: int, seed: int) -> pd.DataFrame:
    if max_samples <= 0 or len(df) <= max_samples:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    sampled, _ = train_test_split(
        df,
        train_size=max_samples,
        random_state=seed,
        stratify=df["label"],
    )
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_and_prepare_dataset(
    config: DatasetConfig,
    data_root: Path,
    seed: int,
) -> pd.DataFrame:
    source = DTI(name=config.name, path=str(data_root))
    raw = source.get_data()
    df = _to_standard_frame(raw)
    df = _binarize_labels(df, config.name, config.threshold)
    df = _stratified_sample(df, config.max_samples, seed)

    if len(df) < 100:
        raise ValueError(f"{config.name}: only {len(df)} rows after preprocessing, need >=100")
    if df["label"].nunique() < 2:
        raise ValueError(f"{config.name}: binary labels collapsed into a single class")

    return df[STANDARD_COLUMNS].copy()


def split_dataset(
    df: pd.DataFrame,
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if val_size <= 0 or test_size <= 0:
        raise ValueError("val_size and test_size must be > 0")
    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be < 1")

    train_df, holdout_df = train_test_split(
        df,
        test_size=val_size + test_size,
        random_state=seed,
        stratify=df["label"],
    )
    relative_test = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=relative_test,
        random_state=seed,
        stratify=holdout_df["label"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_splits(
    dataset_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_root: Path,
) -> dict[str, Path]:
    split_dir = output_root / "splits" / dataset_name.lower()
    split_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "train": split_dir / "train.csv",
        "val": split_dir / "val.csv",
        "test": split_dir / "test.csv",
    }
    train_df.to_csv(paths["train"], index=False)
    val_df.to_csv(paths["val"], index=False)
    test_df.to_csv(paths["test"], index=False)
    return paths


def dataset_profile(dataset_name: str, threshold: float, df: pd.DataFrame) -> dict[str, object]:
    return {
        "dataset": dataset_name,
        "label_type": "continuous affinity + derived binary label",
        "binary_threshold": threshold,
        "rows": int(len(df)),
        "unique_smiles": int(df["smiles"].nunique()),
        "unique_proteins": int(df["protein_sequence"].nunique()),
        "positive_rate": float(df["label"].mean()),
        "affinity_min": float(df["affinity"].min()),
        "affinity_max": float(df["affinity"].max()),
        "affinity_median": float(df["affinity"].median()),
    }
